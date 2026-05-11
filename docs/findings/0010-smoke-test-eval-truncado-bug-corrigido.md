# F-0010 — Smoke test com eval truncado escondia ausência de classes; corrigido com `shuffle_eval`

- **Data:** 2026-05-10
- **Status:** Confirmed (bug detectado, corrigido, validado)
- **Categoria:** metodologia

## O que descobrimos

A primeira execução do smoke test do baseline ResNet-50 produziu **uma matriz de confusão sem sentido** que poderia ter sido confundida com bug do modelo. A causa raiz é a **interação de três decisões corretas em isolamento** que juntas escondem informação crítica:

1. **Manifesto ordenado por classe** (`eda.py` faz `sort_values(["class_4", ...])`) — facilita inspeção mas agrupa classes.
2. **`shuffle=False` em val/test** (ADR-0006, regra de avaliação determinística) — preserva determinismo mas não distribui classes.
3. **`max_batches` em smoke test** (ADR-0008, smoke como sanity check rápido) — limita iterações mas, combinado com (1) e (2), avalia apenas os primeiros batches de val/test que são da mesma classe.

Adicionalmente, o `build_dataloaders` original tinha um único parâmetro `batch_size` aplicado a train/val/test, ignorando `batch_size_eval` da TrainConfig — bug menor que reduzia a janela do smoke ainda mais.

### Sintoma original (smoke run `20260510_205105_resnet50_smoke_smoke`)

Matriz de confusão do test set após 2 epochs:

```
                non_dem  very_mild  mild_or_mo
non_demented      0          0          0      ← n=0 amostras reais
very_mild         0          0          0      ← n=0 amostras reais
mild_or_mod     191         48          1      ← TODAS as 240 amostras
```

Linha non_demented zerada significa que **nenhum exemplo de non_demented apareceu no eval**, embora o test set tenha 13.237 slices, sendo 10.492 (79%) non_demented. O smoke truncou em 30 batches × 8 = 240 amostras, todas vindas do início ordenado do test (3 sujeitos mild_or_moderate primeiro).

Sem a corrigir, balanced_accuracy = 0.004 sugere "modelo quebrado", quando na verdade o modelo estava aprendendo e a métrica é que estava sendo computada sobre uma fatia inválida.

## Correção

`src/data/dataset.py` — `build_dataloaders` agora aceita:

- `batch_size_train` e `batch_size_eval` separados (bug menor corrigido);
- `shuffle_eval` (default `False`) + `shuffle_eval_seed` (default `42`) — quando `True`, embaralha val/test uma vez com seed fixa via `torch.Generator`. Mantém determinismo (mesma seed → mesma ordem) e distribui classes entre batches.

`src/training/run.py` passa `shuffle_eval=cfg.smoke_test` — eval embaralhada apenas no modo smoke; runs reais mantêm ordem determinística natural (sem necessidade, já que percorrem o fold completo).

## Validação pós-correção

Smoke run `20260510_205637_resnet50_smoke_smoke`:

**Distribuição de classes no test set avaliado (480 samples = 30 batches × 16):**

| Classe | Reais | Esperado por proporção global |
|---|---:|---:|
| non_demented | 381 (79,4%) | ~79,3% ✓ |
| very_mild | 83 (17,3%) | ~15,7% ✓ |
| mild_or_moderate | 16 (3,3%) | ~5,1% ✓ |

Distribuição **bate com a proporção global do test** dentro de variação amostral. Eval agora reflete a realidade do fold.

**Matriz de confusão pós-correção:**

```
                non_dem  very_mild  mild_or_mo
non_demented    323        57          1       ← acerta 85% dos non_demented
very_mild        63        20          0       ← acerta 24% dos very_mild
mild_or_mod      11         5          0       ← acerta 0% dos minoritários
```

**Métricas finais no test:**

- `accuracy`: 0,7146 (próximo da fração de non_demented = "trivial baseline")
- `balanced_accuracy`: 0,3629 (próximo de aleatório 0,333)
- `macro_f1`: 0,358
- `auc_macro`: **0,564** — acima de 0,5 indica que o modelo já está discriminando algo, mesmo com argmax favorecendo a classe majoritária.

**Loss curve auditada:**

| Epoch | train_loss | val_loss | val_bal_acc | val_auc_macro |
|---|---:|---:|---:|---:|
| 1 | 1,1092 | 1,0725 | 0,3804 | 0,6288 |
| 2 | 1,0581 | 1,0199 | 0,3333 | 0,6468 |

train_loss e val_loss **diminuem consistentemente**. AUC macro **aumenta** mesmo quando balanced_accuracy cai — isso indica que o modelo está aprendendo a **ordenar** as classes pelas probabilidades, mas o **threshold de decisão** (argmax) ainda favorece non_demented por causa do pouco treino (30 batches). Comportamento esperado em smoke test, não bug.

## Implicação

- **Pipeline está correto.** A validação pós-correção mostra distribuição realista, métricas coerentes e progresso de aprendizado.
- **Lição:** sempre que `max_batches` for usado em eval, **deve vir com shuffle_eval=True**. Se não for, o eval pode mascarar bugs ou interpretar erroneamente o estado do modelo.
- **Para runs reais (sem `--smoke`),** eval percorre o fold inteiro — não precisa shuffle. Comportamento atual (`shuffle_eval=cfg.smoke_test`) está correto.
- **Reprodutibilidade preservada:** `shuffle_eval_seed=42` (igual seed do split) garante que o smoke produz a mesma ordem de batches em qualquer máquina.
- **Não foi necessário mudar ADR-0006** — o princípio de "val/test sem shuffle nas runs reais" continua válido; mudou apenas o smoke. Mantém-se F-0007 como gotcha de runs reais.
- **Conexão com F-0003 (erro do dual-view):** se eu não tivesse parado para analisar a matriz de confusão sem sentido e tivesse simplesmente declarado "smoke OK loss diminui", o bug propagaria para o treino real e horas de Colab seriam perdidas. Reforça que **auditoria visual/numérica** de cada artefato é não-negociável.

## Notas / armadilhas

- `total_batches_per_epoch` no smoke é finito; o weighted CE precisa de **muitas mais iterações** para balancear classes. Em 30 batches × 8 = 240 imagens, mild_or_moderate aparece ~17 vezes — insuficiente para sair do colapso para classe majoritária. **Treino real (Colab) com 1.886 batches/epoch × 20 epochs deve resolver isso**.
- `auc_macro` é uma métrica útil para detectar "modelo aprendeu ranking mas threshold ruim", complementando balanced_accuracy. Manter no relatório (ADR-0005 já cobre).
- `--smoke-batches` no `run.py` (default 50) deve ser **pelo menos batch_size × N grande o bastante para >=1 amostra de cada classe**. Com a classe `mild_or_moderate` representando ~5% do val, precisa de batch_size ≥ ~20 ou ≥ 30 batches × 16 amostras = 480 totais para garantir cobertura — atende com folga.
- Tempo: 30 batches train + 30 batches eval em CPU = ~60s por epoch para ResNet-50. Smoke completo (2 epochs + eval final) = ~2 minutos. Treino real em CPU seria impraticável (~5 dias estimados), confirmando ADR-0003 (Colab para treino).
