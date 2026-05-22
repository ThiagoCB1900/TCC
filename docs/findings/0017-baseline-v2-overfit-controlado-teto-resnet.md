# F-0017 — Baseline V2: overfit controlado, mas teto do ResNet-50 em ~0,59-0,62 balanced_acc

- **Data:** 2026-05-22
- **Status:** Confirmed (2 runs V2 no Kaggle T4, mesmo split_v1 das runs V1)
- **Categoria:** metodologia

## O que descobrimos

As mudanças do V2 (ADR-0010: LR diferenciado, dropout 0.3, weight_decay 0.1, augmentation forte, warmup 2, patience 5) **controlaram o overfit catastrófico do V1** (F-0013), mas **não elevaram o teto de desempenho**. ResNet-50 baseline converge para ~0,59-0,62 balanced_accuracy no test, independente do regime de regularização. O gargalo deixou de ser overfit e passou a ser a capacidade do modelo/representação.

## Evidência

### Curva de treino — V1 vs V2 (weighted)

| | V1 weighted | V2 weighted |
|---|---|---|
| train_loss | colapsa 0,83 → 0,028 em 4 ep | cai gradual 1,07 → 0,40 em 16 ep |
| val_loss | triplica (0,61 → 1,93) | estável (~0,62-0,73) |
| best epoch | **1** (instável) | **11** (saudável) |
| epochs até early stop | 4 | 16 |

Curva V2 weighted (val por epoch): balanced_acc sobe de 0,496 (ep1) e estabiliza em platô ~0,62-0,63 (best 0,6327 na ep11). train_loss desce devagar sem colapsar. **Overfit resolvido.**

### Métricas finais no test (n=13.237, mesmo split_v1)

| Config | best_ep | accuracy | balanced_acc | macro_f1 | auc_macro | F1 mild+mod |
|---|---:|---:|---:|---:|---:|---:|
| V1 weighted | 1 | 0,699 | **0,623** | 0,554 | 0,804 | 0,400 |
| V1 noweight | 4 | 0,672 | 0,497 | 0,454 | 0,786 | 0,166 |
| **V2 weighted** | 11 | 0,681 | **0,587** | 0,513 | 0,804 | 0,323 |
| **V2 noweight** | 9 | 0,750 | 0,546 | 0,529 | 0,821 | 0,376 |

Runs: `20260522_013246_resnet50_v2_class3_weighted_kaggle`, `20260522_042821_resnet50_v2_class3_noweight_kaggle`.

## Interpretação

1. **O 0,623 do V1 weighted é instável / não-reprodutível** — obtido na epoch 1 (modelo quase no prior ImageNet, antes do colapso de overfit). O 0,587 do V2 vem de 11 epochs de treino estável → **número mais confiável, ainda que menor**. Metodologicamente, **V2 é o baseline oficial** apesar do número ligeiramente inferior.

2. **Teto do ResNet-50 baseline ≈ 0,59-0,62 balanced_acc.** V1 e V2 convergem para essa faixa. Regularização estabilizou mas não elevou o teto.

3. **AUC macro ~0,80-0,82 em todas as configs.** O modelo aprende um ranking decente das classes; a limitação está na separabilidade no limiar de decisão, não na ausência de sinal.

4. **Nuance weighted vs noweight no V2:** com augmentation forte + dropout, o noweight não colapsa como no V1. V2 noweight tem macro_f1 (0,529) e F1 da minoritária (0,376) ligeiramente melhores que V2 weighted (0,513 / 0,323), mas **V2 weighted tem balanced_accuracy maior (0,587 vs 0,546)** — e balanced_acc é a métrica primária (ADR-0005). Weighted segue como configuração principal, mas a diferença encolheu (a regularização do V2 já mitiga parte do desbalanceamento).

5. **O gargalo mudou.** Não é mais overfit (resolvido). É a representação: fatia axial 2D única + ResNet-50 + split por paciente. Para subir o teto, a alavanca é **dados** (2.5D — contexto de fatias vizinhas, ADR-0011) ou **arquitetura** (ViT/Swin com atenção global) — não mais hiperparâmetro de regularização.

## Implicação

- **Baseline ResNet-50 está FECHADO.** V2 weighted é o baseline oficial (treino estável, número confiável, metodologia defensável). Teto estabelecido: ~0,59 balanced_acc / 0,80 AUC.
- **Hiperparâmetros de regularização foram exauridos** como alavanca (V1 vs V2 cobre o espectro under→over regularizado).
- **Próximo passo recomendado:** partir para **ViT-Base/16 e Swin-T** (objetivo central do TCC) com a mesma metodologia e mesmo split_v1. Se ViT/Swin também platôarem em ~0,60, então 2.5D vira a alavanca de dados a aplicar a todos os modelos.
- Para comparação ResNet vs ViT vs Swin via **McNemar** (ADR-0005), será preciso salvar as predições por amostra (`y_pred` no test) — adicionar essa persistência ao loop antes dos treinos de ViT/Swin.

## Notas / armadilhas

- **Não reportar o 0,623 do V1 como "melhor resultado".** É instável (epoch 1). Reportar V2 weighted (0,587) como baseline, explicando que o V1 colapsava por overfit (F-0013) — a narrativa honesta é "estabilizamos o treino; o teto do baseline é ~0,59".
- **McNemar entre V1 e V2 não foi feito** — exigiria predições por amostra, que não foram salvas nessas runs. Não afirmar significância estatística da diferença V1 vs V2; tratá-la como dentro da variância de treino.
- **A queda val→test é consistente** (~0,05 em ambas versões), indicando que o val (52 pacientes) é um proxy razoável do test (53 pacientes). Split saudável.
- Tempo: V2 weighted levou ~2h54 (10.439 s) para 16 epochs no T4; noweight ~2h27. Dentro da cota Kaggle.
