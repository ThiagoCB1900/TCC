# F-0013 — Baseline ResNet-50 V1: weighted CE empiricamente validado, overfit catastrófico em 4 epochs

- **Data:** 2026-05-17
- **Status:** Confirmed (2 runs no Kaggle T4 com a mesma seed e split)
- **Categoria:** metodologia

## O que descobrimos

O primeiro par de runs reais do baseline ResNet-50 (configuração ADR-0008) entregou dois sinais simultaneamente:

1. **Validação empírica do ADR-0007** (weighted CE com pesos `'balanced'`): com peso, balanced_accuracy +24% e F1 da classe minoritária +135% em relação à ablação sem peso.
2. **Overfit catastrófico**: train_loss colapsa ~30× em 4 epochs enquanto val_loss triplica — modelo memoriza o train antes de generalizar.

Os 2 resultados são consistentes entre si e indicam que a metodologia metódica está correta (weighted loss funciona como previsto), mas os hiperparâmetros de fine-tuning estão mal calibrados para o tamanho efetivo do nosso dataset (~242 pacientes train).

## Evidência

### Tabela comparativa — runs do baseline V1

| Run ID | Loss | Epoch best | Test acc | Test bal_acc | Test macro_F1 | Test AUC | Tempo |
|---|---|---:|---:|---:|---:|---:|---:|
| `20260517_201801_resnet50_class3_weighted_kaggle` | Weighted CE `'balanced'` | 1/4 | 0,699 | **0,623** | **0,554** | 0,804 | 52 min |
| `20260517_214128_resnet50_class3_noweight_kaggle` | CE plana | 4/7 | 0,672 | 0,497 | 0,454 | 0,786 | 90 min |

### Métricas por classe (test set, 13.237 slices)

| Classe | Suporte | Com peso (P / R / F1) | Sem peso (P / R / F1) |
|---|---:|:---:|:---:|
| non_demented | 10.492 | 0,965 / 0,709 / 0,818 | 0,926 / 0,713 / 0,806 |
| very_mild | 2.074 | 0,319 / 0,731 / 0,444 | 0,282 / 0,633 / 0,391 |
| **mild_or_moderate** | 671 | **0,375 / 0,429 / 0,400** | **0,193 / 0,146 / 0,166** |

### Curva de loss da run com peso (evidência do overfit)

```
epoch │ train_loss │ val_loss │ val_bal_acc │ val_AUC
   1  │   0,829    │   0,607  │   0,680     │  0,629  ← best
   2  │   0,254    │   1,017  │   0,534     │  ↑
   3  │   0,064    │   1,734  │   0,512     │  ↑
   4  │   0,028    │   1,926  │   0,474     │  ↑
            ↓                       ↓
       cai 30×                  perde 30%
```

train_loss em ~0,028 na epoch 4 indica que o modelo está praticamente certo em **todos** os exemplos de treino — memorização. val_loss subindo ao mesmo tempo prova que essa "certeza" não se transfere para dados não vistos.

### Matrizes de confusão (test)

**Com peso:**
```
                pred:  non_dem  very_mild  mild_or_mod
real non_demented      7442     2873        177      (R=0,71)
real very_mild          255     1516        303      (R=0,73)
real mild_or_moderate    14      369        288      (R=0,43)
```

**Sem peso:**
```
                pred:  non_dem  very_mild  mild_or_mod
real non_demented      7483     2786        223      (R=0,71)
real very_mild          575     1313        186      (R=0,63)
real mild_or_moderate    24      549         98      (R=0,15)
```

Sem peso, o modelo confunde maciçamente `mild_or_moderate` com `very_mild` (549/671 = 82% dos erros vão para very_mild). Com peso, essa confusão cai para 369/671 = 55%, e o modelo passa a acertar 288/671 = 43% das amostras minoritárias.

## Implicação

### O que está validado

- **ADR-0001 (3 classes)**: as 3 classes têm comportamento distinto na matriz de confusão; agrupar Mild+Moderate não destruiu informação.
- **ADR-0002 (split por paciente)**: sem contaminação intra-paciente; AUC 0,80 é honesto.
- **ADR-0005 (métricas balanceadas)**: accuracy bruta de 0,70 seria armadilhosamente próxima do trivial (~0,78 se predisse sempre non_demented). Balanced accuracy e macro-F1 expõem o ganho real.
- **ADR-0007 (weighted CE)**: ganho de 24% em balanced_acc e 135% em F1 da minoritária quantifica o impacto. Material direto para tese.
- **Metodologia distingue-se claramente do RanCom-ViT** (F-0008): nosso AUC 0,80 honesto vs accuracy 0,99 inflada por leakage.

### O que precisa melhorar (raiz do overfit)

| # | Causa provável | Evidência | Correção proposta no ADR-0010 |
|---|---|---|---|
| 1 | LR alto (1e-4) para fine-tune full | train_loss colapsa 30× em 4 epochs | LR diferenciado: backbone 1e-5, head 1e-4 |
| 2 | Backbone descongelado (25M params vs 242 pacientes efetivos) | Ratio params/dados independentes ~100k:1 | LR baixo no backbone (acima); considerar linear probing em ablação |
| 3 | Augmentation tímida demais (rotação ±5°, jitter 0,1) | Não foi suficiente para prevenir memorização | Aumentar para rotação ±15°, jitter 0,2, adicionar pequena translação |
| 4 | Sem dropout no classifier head (timm default) | Head com 2048→3 sem regularização | `drop_rate=0.3` |
| 5 | Weight decay padrão (0,05) | Razoável mas pode ajudar mais | Subir para 0,1 |

### Comparação contextual — onde nosso 0,62 se posiciona

| Trabalho | Tarefa | Split | Métrica primária |
|---|---|---|---:|
| Baseline V1 nosso | OASIS-1 T1 3-classes | **por paciente** | bal_acc 0,62 |
| RanCom-ViT (F-0008) | OASIS-1 T1 4-classes | por slice (leakage) | "acc 0,995" (inflado) |
| Leveraging Swin — Swin+LoRA | ADNI dMRI binário (AD vs CN) | por paciente, 5-fold | bal_acc 0,95 |
| Leveraging Swin — ResNet baseline | ADNI dMRI binário (AD vs CN) | por paciente, 5-fold | bal_acc 0,90 |
| Trivial (sempre majoritária) | OASIS-1 T1 3-classes | qualquer | bal_acc 0,33 |

Nossa tarefa é estruturalmente mais difícil (3 classes vs binário; T1 vs dMRI rico; minoritária com 23 pacientes vs ADNI com centenas). Mesmo assim, 0,62 está **bem abaixo** do que ResNet-50 deveria entregar com hiperparâmetros bem calibrados. Esperamos baseline V2 em 0,68-0,75.

## Notas / armadilhas

- **Early stopping pegou bem o overfit** (parou em epoch 4 com weighted; epoch 7 sem peso). Sem ele, val_bal_acc continuaria caindo enquanto train_loss → 0. Patience=3 está calibrado corretamente.
- **Best epoch=1 (com peso)** é sinal vermelho — o modelo "encontrou" sua melhor generalização imediatamente e perdeu daí pra frente. Indica que estamos partindo de pesos pré-treinados muito bons e o fine-tune está degradando. Reforça hipótese de **LR alto demais para o backbone**.
- **AUC macro estável** mesmo durante o overfit em balanced_acc é informativo: o modelo continua aprendendo a **ordenar** as classes (saída softmax discrimina), mas o **limiar de decisão** (argmax) perde calibração. Sugere que **temperature scaling pós-hoc** poderia melhorar balanced_acc sem re-treinar. Anotado para futuro.
- **Diferença de tempo (52 vs 90 min)** vem do early stopping em momentos diferentes — não é instabilidade do hardware.
- **Não tentar comparar diretamente** nossos 0,62 com a literatura ADNI/dMRI binária: tarefas e modalidades diferentes. Comparação justa só com runs no mesmo dataset OASIS-1 com split por paciente — atualmente o único é o **nosso próprio** (RanCom-ViT não conta por F-0008).
