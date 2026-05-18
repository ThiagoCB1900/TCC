# ADR-0010 — Baseline V2: hiperparâmetros revisitados para combater overfit (LR diferenciado, dropout, augmentation forte)

- **Data:** 2026-05-17
- **Status:** Accepted (refinamento do ADR-0008 — não invalida)
- **Decisores:** Thiago, Claude

## Contexto

[F-0013](../findings/0013-baseline-v1-overfit-rapido-weighted-validado.md) documentou o resultado do baseline V1: **weighted CE empiricamente validado** (balanced_acc +24%, F1 minoritária +135% vs sem peso) **mas overfit catastrófico** em 4 epochs (train_loss 30× menor; val_loss 3× maior). Best epoch foi a **epoch 1** — o modelo "encontrou" o ótimo de generalização imediatamente e perdeu daí.

Diagnóstico: hiperparâmetros do ADR-0008 (LR=1e-4 uniforme, drop_rate=0, augmentation tímida, weight_decay=0,05) estão calibrados para datasets maiores. Com 242 pacientes efetivos no train e 25M parâmetros descongelados, o ratio é ~100k:1 — receita para memorização.

ADR-0008 fica como referência da abordagem mínima. Este ADR define **Baseline V2** com mudanças coordenadas para atacar o overfit.

## Alternativas consideradas

### Para o learning rate

- **A** — LR uniforme menor (5e-5). Reduz overfit mas head também aprende mais devagar.
- **B** — **LR diferenciado por grupo**: backbone com LR baixo (1e-5), classifier head com LR alto (1e-4). Reconhece que o head (re-inicializado aleatório) precisa aprender do zero enquanto backbone (pretrained ImageNet) só precisa ajustar fino.
- **C** — Linear probing puro (backbone frozen). Elimina overfit do backbone, mas perde capacidade de adaptação ao domínio médico.

### Para regularização

- **D** — Dropout no head (`drop_rate=0.3`).
- **E** — Weight decay maior (0,1 vs 0,05).
- **F** — DropPath (stochastic depth) no backbone. Disponível em ViT/Swin, não nativo em ResNet — pular.
- **G** — Mixup / Cutmix nos batches de treino. Forte regularização, mas distorce labels em problema multi-classe (suaviza muito a fronteira `mild_or_moderate`).

### Para augmentation

- **H** — Aumentar parâmetros atuais: rotação ±15° (era ±5°), jitter 0,2 (era 0,1).
- **I** — Adicionar `RandomAffine` translação leve (~5% da imagem).
- **J** — Adicionar `RandomErasing` (cutout). Borra partes da imagem.
- **K** — Mudança para pipeline 2.5D (3 slices vizinhos como canais RGB). Mudança maior — fica como **trabalho separado** (V3 se necessário) por escopo.

## Decisão — Baseline V2

Adotar **B + D + E + H + I** em combinação coordenada. Justificativa de cada uma:

| # | Mudança V1 → V2 | Por que |
|---|---|---|
| 1 | **LR diferenciado**: backbone 1e-5, head 1e-4 (era 1e-4 uniforme) | Backbone pretrained só precisa ajuste fino; head reinicializado precisa aprender do zero. Razão 10× é prática comum em fine-tune de CNN/ViT em medical imaging. |
| 2 | **`drop_rate=0.3`** no classifier head (era 0.0) | Regulariza a única camada com pesos aleatórios — onde o overfit começa. |
| 3 | **`weight_decay=0.1`** (era 0,05) | Penaliza magnitudes altas; ataca memorização. |
| 4 | **Augmentation forte**: rotação ±15°, ColorJitter 0,2/0,2, RandomAffine translação (±5%) | Mais variação visual por epoch → modelo vê "exemplos diferentes" mesmo memorizando o train. |
| 5 | **`patience=5`** no early stopping (era 3) | Com LR menor, convergência é mais lenta. Dar margem para encontrar melhor epoch. |
| 6 | **`warmup_epochs=2`** (era 1) | LR diferenciado precisa de aquecimento mais cuidadoso. |

**O que NÃO muda** (continua igual ao ADR-0008):
- Split, dataset, classes, métricas, weighted loss, image_size, batch_size, optimizer (AdamW), scheduler (cosine).

**O que fica fora do escopo de V2** (vai para V3 se necessário):
- Linear probing puro (C) — manter como ablação opcional na semana 5-6.
- Mixup/Cutmix (G) — útil mas adiciona variável extra; resolver overfit primeiro.
- 2.5D (K) — mudança estrutural no `OASISDataset`; fica isolada num V3 dedicado se hiperparâmetros V2 não bastarem.
- DropPath (F) — disponível em ViT, vai entrar quando treinarmos ViT-Base.

## Especificação concreta

### Em `src/training/config.py` — `TrainConfig`

| Campo (novo ou alterado) | Default V2 | Notes |
|---|---|---|
| `lr_head` | `1e-4` | NEW. LR do classifier head |
| `lr_backbone` | `1e-5` | NEW. LR do backbone (ImageNet pretrained) |
| `drop_rate` | `0.3` | NEW. Passado ao `timm.create_model` |
| `weight_decay` | `0.1` | mudou de 0,05 |
| `warmup_epochs` | `2` | mudou de 1 |
| `early_stopping_patience` | `5` | mudou de 3 |
| `augment_strength` | `"strong"` | NEW. Substitui flag única; aceita `"light"` (V1) ou `"strong"` (V2) |

`lr` antigo (uniforme) **continua existente** como fallback — se `lr_head` e `lr_backbone` estiverem ambos `None`, usa `lr` para todos os parâmetros (compatível com V1).

### Em `src/models/resnet.py` — `build_resnet50`

Aceita `drop_rate: float = 0.0` e passa direto pro `timm.create_model`. Default fica em 0 para preservar comportamento V1.

### Em `src/training/loop.py` — `fit`

Quando `lr_head != lr_backbone`, constrói `param_groups`:

```python
head_params, backbone_params = [], []
for name, p in model.named_parameters():
    if not p.requires_grad:
        continue
    if name.startswith(('fc', 'classifier', 'head')):
        head_params.append(p)
    else:
        backbone_params.append(p)
optimizer = AdamW(
    [{"params": backbone_params, "lr": cfg.lr_backbone},
     {"params": head_params, "lr": cfg.lr_head}],
    weight_decay=cfg.weight_decay,
)
```

### Em `src/data/dataset.py` — `build_transform`

`augment_strength` aceita `"light"` (V1, default por retrocompatibilidade) ou `"strong"` (V2):

```python
if augment_strength == "strong":
    steps.extend([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15, fill=0),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=0),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ])
```

## Consequências

- **Risco controlado:** todas as mudanças têm fundamentação direta na literatura de fine-tune CNN médica (ver referências). Cada uma é revisivel isoladamente em ablação se V2 decepcionar.
- **Custo computacional:** ~igual ao V1 (~50-90 min por run no T4). Total adicional: 2 runs (V2 com peso + ablação V2 sem peso) = ~3h de Kaggle. Cabe na cota.
- **Compatibilidade retroativa preservada:** se `lr_head=lr_backbone=null` e `augment_strength='light'`, comportamento V1 idêntico. Não quebra reprodutibilidade do baseline V1.
- **Ablação obrigatória após V2:** comparar V1 com peso vs V2 com peso (ambos com weighted loss). Esperamos balanced_accuracy subindo de 0,62 para 0,68-0,75. Se ficar igual ou pior, revisamos cada mudança em ablation por separação.
- **V3 fica desenhada:** se V2 não bastar, próximo passo é pipeline 2.5D (3 slices vizinhos como canais RGB) — atacar o uso desperdiçado dos 3 canais ImageNet em vez dos hiperparâmetros.
- **Não muda ADR-0008** (framework PyTorch puro): continua válido. ADR-0010 só refina os defaults.
- **Não muda ADR-0007** (weighted CE): continua sendo o tratamento de desbalanceamento.

## Referências

- F-0013 (resultados V1 que motivaram V2)
- ADR-0006 (augmentations leves) — augmentation strong é uma evolução, não substituição
- ADR-0007 (weighted CE)
- ADR-0008 (framework + hiperparâmetros V1)
- **Fine-tuning literature em medical imaging:**
  - Smith, 2017 — *Cyclical Learning Rates* (motivação para LR diferenciado por grupo)
  - Howard & Ruder, 2018 — *Universal Language Model Fine-tuning* (ULMFiT) — popularizou LR diferenciado por camada/grupo
  - Esteva et al., 2017 — *Dermatologist-level classification* — pretrained ImageNet + fine-tune full com LR baixo + dropout forte
  - Leveraging Swin (paper já em `docs/Papers/`): usa LoRA, que conceitualmente faz LR diferenciado extremo (=0 no backbone, full nas adapters) — convergente com nossa direção
