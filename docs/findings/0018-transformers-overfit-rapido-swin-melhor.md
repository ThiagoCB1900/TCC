# F-0018 — Transformers (ViT/Swin) overfittam rápido com hiperparâmetros V2; Swin-T é o melhor modelo

- **Data:** 2026-05-23
- **Status:** Confirmed (ViT-Base e Swin-T weighted no Kaggle T4, mesmo split_v1)
- **Categoria:** metodologia

## O que descobrimos

Os Vision Transformers treinados com os hiperparâmetros V2 (ADR-0010, calibrados para o ResNet-50) **voltam a sofrer overfit catastrófico** — best epoch 2, train_loss colapsa, val_loss explode. Apesar disso, **Swin-T é o melhor modelo até agora** (test balanced_acc 0,594), superando ResNet-50 V2 (0,587) e ViT-Base (0,521), e é estatisticamente superior ao ViT (McNemar p=0,017). O val pico do Swin (0,668 na epoch 2) é o melhor já observado — indicando ganho preso atrás do overfit.

## Evidência

### Curvas de treino (overfit explícito)

**ViT-Base/16 weighted:**
```
ep  train   val_loss  bal_acc
 1  0,8137  0,5964    0,5347
 2  0,3765  1,0830    0,5407  <- best
 3  0,1750  1,6651    0,4876
 ...
 7  0,0438  3,0402    0,4974   (train colapsa, val explode)
```

**Swin-T weighted:**
```
ep  train   val_loss  bal_acc
 1  0,8470  0,7033    0,6662
 2  0,4906  0,7845    0,6681  <- best (melhor val de todo o projeto)
 3  0,2672  1,0932    0,6002
 ...
 7  0,0520  3,0137    0,4313   (train colapsa, val explode)
```

Mesmo padrão do ResNet V1 (F-0013): memorização rápida. Os hiperparâmetros V2 que estabilizaram o ResNet (LR backbone 1e-5, drop_path 0.1, weight_decay 0.1) **são insuficientes para transformers**, mais propensos a overfit — especialmente ViT-Base (86M params vs 28M do Swin-T vs 23M do ResNet).

### Comparativo final no test (n=13.237, mesmo split_v1, weighted)

| Modelo | params | accuracy | balanced_acc | macro_f1 | auc_macro | best_ep |
|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 V2 | 23,5M | 0,681 | 0,587 | 0,513 | 0,804 | 11 |
| ViT-Base/16 | 85,8M | 0,730 | 0,521 | 0,483 | **0,822** | 2 |
| **Swin-T** | 27,5M | 0,723 | **0,594** | **0,528** | **0,823** | 2 |

### McNemar (predições por amostra, mesmo test)

- **ViT vs Swin: b=770, c=868, p=0,0165 → significativo.** Swin acerta mais amostras que ViT erra do que o contrário. Combinado com balanced_acc, **Swin > ViT com significância estatística**.
- ResNet V2 não entrou no McNemar: foi treinado antes da persistência de predições (`test_predictions.npz`). Para incluí-lo, re-rodar/re-avaliar o checkpoint.

## Interpretação

1. **Swin-T é o melhor modelo do projeto** (bal_acc 0,594, macro_f1 0,528, AUC 0,823). Supera o baseline ResNet, mas por margem pequena sobre o ResNet (+0,007 bal_acc) — provavelmente não significativa (confirmar com McNemar quando ResNet tiver predições).
2. **ViT-Base é o pior em balanced_acc** (0,521) apesar da maior accuracy (0,730): polariza para a classe majoritária. Overfit mais severo (modelo grande, dados pequenos). É o caso clássico de ViT precisar de muito mais dado ou muito mais regularização.
3. **AUC dos transformers (0,822-0,823) > ResNet (0,804)**: transformers aprendem um ranking melhor das classes; o overfit corrompe o limiar de decisão (argmax), derrubando balanced_acc. Sugere que **controlar o overfit deve destravar ganho real**.
4. **Val pico do Swin = 0,668** (melhor de todo o projeto, vs 0,633 do ResNet V2). Há desempenho preso atrás do overfit precoce.
5. **Teto persistente** ~0,59-0,67 val balanced_acc em todas as arquiteturas → o gargalo dos DADOS (fatia axial 2D única) é real, mas **a regularização insuficiente dos transformers é um gargalo adicional removível**.

## Implicação — duas alavancas, nesta ordem

1. **Regularização específica para transformers ("V3-transformer")** — dinheiro na mesa, pois os transformers estão claramente overfittando e o Swin já mostra val 0,668. Ajustes propostos (análogos ao V1→V2 do ResNet):
   - `lr_backbone` menor: 5e-6 ou 2e-6 (vs 1e-5).
   - `drop_path_rate` maior: 0,2-0,3 (vs 0,1) — stochastic depth é a regularização-chave de transformers.
   - `weight_decay` maior: 0,2-0,3 (vs 0,1) — ViT original usa 0,3.
   - `warmup_epochs` maior: 3-5 (vs 2).
   - (avançado, se necessário) layer-wise LR decay (LLRD).
   Todos já expostos via CLI (`--lr-backbone`, `--drop-path-rate`, `--weight-decay`, `--warmup-epochs`) — não exige mudança de código, só novos comandos no notebook 04.
2. **2.5D (ADR-0011)** — ataca o gargalo de dados; aplicar ao melhor modelo depois de estabilizado.

## Notas / armadilhas

- **Não reportar o ViT como "transformers são piores que CNN".** O ViT está subtreinado/overfittado, não esgotado. A conclusão honesta é "ViT-Base exige regularização forte neste regime de dados pequenos; sem ela, overfitta antes de generalizar".
- **best_epoch=2 + early stopping patience=5** → rodou 7 epochs. O early stopping pegou o melhor ponto corretamente; o problema é que o melhor ponto é precoce (pré-overfit).
- **ResNet sem predições salvas** — re-rodar ResNet V2 (agora com persistência) para McNemar completo das 3 arquiteturas é recomendável antes da escrita final.
- **Tempos:** ViT 4h20 (15.610s, 86M params), Swin 1h40 (6.050s). ViT é caro — considerar isso ao planejar as rodadas de tuning.
