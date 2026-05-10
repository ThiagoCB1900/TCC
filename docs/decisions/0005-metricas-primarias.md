# ADR-0005 — Métricas primárias: macro-F1, balanced accuracy, AUC

- **Data:** 2026-05-10
- **Status:** Accepted
- **Decisores:** Thiago, Claude

## Contexto

Mesmo no esquema de 3 classes (ver ADR-0001), o desbalanceamento permanece severo: ~12× entre `non_demented` (266 pacientes) e `mild_or_moderate` (23). Em slices, fica ainda mais extremo (77,8% vs 6,4%). Reportar **acurácia bruta** seria enganoso — um classificador trivial que sempre prediz `non_demented` já bate ~78% (ver F-0002 para distribuição exata).

Esse exato erro é o que limita a interpretabilidade dos resultados do notebook CNN comparativo (F-0005).

## Alternativas consideradas

- **A — Acurácia simples.** Inadequada por causa do desbalanceamento.
- **B — F1 por classe + macro-F1.** Mostra performance individual em cada classe; macro-F1 dá média não ponderada (não favorece classe majoritária).
- **C — Balanced accuracy.** Média da recall por classe — equivalente a tratar classes como se fossem balanceadas.
- **D — AUC (ROC) por classe (one-vs-rest) + AUC macro.** Métrica padrão em diagnóstico médico; independente do limiar de decisão.
- **E — McNemar para comparação par-a-par entre modelos.** Teste estatístico que verifica se a diferença entre dois classificadores no mesmo test set é significativa.

## Decisão

Reportar **B + C + D + E em conjunto**, com a seguinte hierarquia:

1. **Macro-F1** como métrica primária para o leaderboard interno (ResNet vs ViT vs Swin).
2. **Balanced accuracy** como segunda métrica resumo.
3. **AUC macro (one-vs-rest)** como métrica clínica (independente de limiar).
4. **F1 por classe + matriz de confusão** sempre reportadas no apêndice.
5. **McNemar** para validar que diferenças entre modelos são estatisticamente significativas (regra crítica do plano de 60 dias).

Acurácia bruta pode ser reportada para comparabilidade com outros papers, mas **nunca como métrica primária** e sempre acompanhada de balanced accuracy para contexto.

Implementação em `src/evaluation/metrics.py` usando `torchmetrics` e `sklearn.metrics`. Bootstrap (1000 amostras) para intervalos de confiança 95% nas métricas reportadas.

## Consequências

- O TCC pode citar acurácia X de papers anteriores e contrastar mostrando macro-F1 ou balanced accuracy nesses mesmos resultados (quando reportados) — exposição honesta da diferença.
- Permite defender resultados modestos em macro-F1 contra críticas baseadas em "mas a acurácia foi alta": acurácia alta com macro-F1 baixa = modelo trivial.
- Tabela final do TCC vai ter ~5 colunas de métricas + 1 de p-valor McNemar — denso mas defensável.
- Bootstrap adiciona algum custo computacional na fase de avaliação; cabe nas semanas 4-6.

## Referências

- Findings: F-0002 (distribuição), F-0005 (notebook CNN só reporta accuracy)
- CLAUDE.md, "Decisões obrigatórias", "Métricas primárias"
- torchmetrics docs: https://lightning.ai/docs/torchmetrics/
