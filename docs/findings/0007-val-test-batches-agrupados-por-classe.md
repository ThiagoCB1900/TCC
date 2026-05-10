# F-0007 — Batches de val/test sem shuffle ficam agrupados por classe (não é bug)

- **Data:** 2026-05-10
- **Status:** Confirmed (comportamento intencional)
- **Categoria:** metodologia

## O que descobrimos

Durante inspeção do `OASISDataset` em `src/data/inspect_dataset.py`, o primeiro batch do `val_loader` (batch_size=8, `shuffle=False`) retornou todos os 8 rótulos iguais a `2` (`mild_or_moderate`).

Isso **não é bug**: o manifesto (gerado pela EDA) ordena os arquivos por `(class_4, subject, mpr, slice_idx)`. Quando o split é aplicado e o DataLoader não embaralha, os primeiros batches do val/test contêm apenas uma classe — exatamente a primeira classe do manifesto que aparece nos sujeitos daquele fold.

`val/test` propositalmente **não embaralham** (regra de avaliação determinística — duas execuções consecutivas devem produzir métricas idênticas). Embaralhar um fold de avaliação introduziria não-determinismo cosmético sem benefício.

## Evidência

```
[val] 12871 slices, 52 pacientes, label_scheme=class_3
=== Estatisticas de 1 batch (val) ===
  x.shape = (8, 3, 224, 224), dtype=torch.float32
  y.shape = (8,), dtype=torch.int64
  y unique = [2]                     # <-- só mild_or_moderate
```

Em contraste, train_loader (com `shuffle=True`):
```
=== Estatisticas de 1 batch (train) ===
  y unique = [0, 1, 2]               # <-- todas as 3 classes
```

## Implicação

- **Não afeta métricas finais.** Macro-F1, balanced accuracy, AUC são calculadas sobre o conjunto inteiro de previsões, independentemente da ordem de processamento.
- **Afeta inspeção por batch durante validação.** Se alguém olhar `loss` ou `accuracy` por batch sequencialmente durante eval, vai ver oscilações artificiais por causa da composição de classes em cada batch.
- **Afeta sanity-check trivial** "checar se o modelo está vendo todas as classes" via `next(iter(val_loader))` — pode dar falso negativo. Para esse tipo de checagem, usar `train_loader` ou iterar várias vezes.
- **Possíveis ajustes futuros (não aplicados agora):**
  1. Embaralhar val/test com seed fixa uma única vez no construtor → mantém determinismo, distribui classes nos batches.
  2. Manter como está e documentar (escolha atual).
  3. Reordenar o manifesto por slice_idx ao invés de por classe — afeta a EDA também.

## Notas / armadilhas

- `model.eval()` + agregação de previsões em todo o fold é o padrão correto. Eval-on-batch isolado é aproximação debug-only.
- Se o aluno (ou Claude futuro) implementar uma callback de "amostrar batch durante validação para inspeção", **forçar shuffle nessa amostragem** ou pular para um índice aleatório.
- Comportamento documentado também no docstring de `src/data/dataset.py` na função `build_dataloaders`.
