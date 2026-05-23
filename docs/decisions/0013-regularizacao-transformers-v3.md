# ADR-0013 — Regularização forte para transformers ("V3-transformer")

- **Data:** 2026-05-23
- **Status:** Accepted (refina ADR-0010/ADR-0012 para o caso transformer)
- **Decisores:** Thiago, Claude

## Contexto

F-0018 mostrou que ViT-Base e Swin-T, com os hiperparâmetros V2 (calibrados para o ResNet), **overfittam rápido** (best epoch 2; train_loss colapsa; val_loss explode). O Swin já é o melhor modelo (test bal_acc 0,594) e seu val pico foi 0,668 — há ganho preso atrás do overfit. Transformers, especialmente ViT-Base (86M params), são mais propensos a memorizar em datasets pequenos (242 pacientes train) e exigem regularização mais forte que CNNs.

## Decisão

Re-treinar ViT-Base e Swin-T com um conjunto de regularização mais forte ("V3-transformer"), mantendo todo o resto do pipeline (split, weighted CE, augmentation strong, métricas). **Tudo via flags CLI já existentes — sem mudança de código.**

| Hiperparâmetro | V2 (ResNet) | **V3-transformer** | Justificativa |
|---|---|---|---|
| `--lr-backbone` | 1e-5 | **5e-6** | backbone transformer pretrained precisa de LR muito baixo no fine-tune |
| `--lr-head` | 1e-4 | 1e-4 (mantém) | head reinicializado aprende rápido |
| `--drop-path-rate` | 0.1 | **0.2** | stochastic depth é a regularização-chave de transformers |
| `--weight-decay` | 0.1 | **0.2** | ViT original usa 0.3; 0.2 é meio-termo |
| `--warmup-epochs` | 2 | **3** | transformers são sensíveis a warmup |
| `--patience` | 5 | 5 (mantém) | com LR menor, treina mais devagar; patience dá margem |
| `--epochs` | 20 | 20 (mantém) | early stopping corta antes se necessário |

Comandos (no notebook 04):
```bash
python -m src.training.run --model vit_base_16 --epochs 20 \
  --lr-backbone 5e-6 --drop-path-rate 0.2 --weight-decay 0.2 --warmup-epochs 3 \
  --batch-size 32 --batch-size-eval 64 --num-workers 2 \
  --run-name vit_base_16_v3_class3_weighted_kaggle

python -m src.training.run --model swin_tiny --epochs 20 \
  --lr-backbone 5e-6 --drop-path-rate 0.2 --weight-decay 0.2 --warmup-epochs 3 \
  --batch-size 32 --batch-size-eval 64 --num-workers 2 \
  --run-name swin_tiny_v3_class3_weighted_kaggle
```

## Alternativas consideradas

- **Layer-wise LR decay (LLRD)** — LR decai por profundidade de camada; padrão-ouro para fine-tune de transformers. **Não adotado agora** porque exige código novo (param groups por camada). Reservado se V3 não bastar.
- **Congelar backbone (linear probing)** — elimina overfit do backbone mas perde adaptação ao domínio. Ablação possível, não primeira escolha.
- **Augmentation ainda mais forte (mixup/cutmix)** — reservado; resolver via LR/drop_path/wd primeiro (menos variáveis).
- **Reduzir para ViT-Small/DeiT** — trocaria o modelo-âncora do TCC; manter ViT-Base e regularizar.

## Consequências

- **Critério de sucesso:** best epoch sobe de 2 para algo como 5-12 (treino mais longo e estável); val_loss para de explodir; test balanced_acc sobe acima de 0,594 (Swin) e 0,521 (ViT).
- Se V3 destravar o Swin (esperado, dado val pico 0,668), ele consolida como melhor modelo e candidato para 2.5D (ADR-0011) e interpretabilidade.
- Se o ViT-Base continuar overfittando mesmo com V3, é evidência de que 86M params é demais para este dataset — conclusão honesta e defensável (ViT precisa de mais dados; Swin hierárquico generaliza melhor aqui).
- Custo: 2 runs no Kaggle (~4h ViT + ~2h Swin). Cabe na cota.
- **Não muda os defaults do código** (continuam V2, bons para ResNet). V3 é aplicado por flag — preserva reprodutibilidade do baseline.

## Referências

- F-0018 (overfit dos transformers), F-0013/F-0017 (paralelo com o V1→V2 do ResNet).
- ADR-0010 (V2 hiperparâmetros), ADR-0012 (modelos), ADR-0011 (2.5D reservado).
- Dosovitskiy et al. 2020 (ViT — weight decay 0.3, forte regularização); Liu et al. 2021 (Swin — drop_path por estágio).
