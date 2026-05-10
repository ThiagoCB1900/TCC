# ADR-0001 — Esquema de classes: 3 classes (Mild+Moderate fundidos)

- **Data:** 2026-05-10
- **Status:** Accepted (a confirmar com orientador)
- **Decisores:** Thiago, Claude

## Contexto

A versão Kaggle do OASIS-1 entrega 4 rótulos pré-classificados por CDR: Non Demented, Very Mild, Mild, Moderate. O plano original do TCC fixava classificação binária (CDR=0 vs CDR>0). A EDA executada em 2026-05-10 (ver F-0002) revelou que **Moderate Dementia tem apenas 2 pacientes únicos** — split estratificado por subject ID em treino/val/test seria matematicamente inviável para essa classe (test set teria 0 ou 1 paciente).

## Alternativas consideradas

- **A — Manter as 4 classes do Kaggle.** Risco alto: test set de Moderate fica com 0-1 paciente, métricas por classe ficam instáveis, McNemar perde sentido. Difícil de defender na banca.
- **B — Binário (Demented vs Non Demented).** Funde Mild+Moderate+Very Mild em uma classe (81 pacientes) vs Non Demented (266). Mais simples, alinhado ao plano original do CLAUDE.md. Perde granularidade clinicamente útil de "muito leve" vs "leve+moderada".
- **C — 3 classes: Non Demented / Very Mild / Mild+Moderate.** Funde só Mild (21) com Moderate (2) → 23 pacientes. Mantém Very Mild (58) separado. Granularidade clínica preservada onde os dados permitem; classe "Mild+Moderate" agrupa estágios já clinicamente diagnosticados (CDR ≥ 1).

## Decisão

Optamos por **C — 3 classes** com `non_demented` (266 pac.) / `very_mild` (58) / `mild_or_moderate` (23). Granularidade clínica é preservada na fronteira mais relevante (estágio pré-clínico vs clínico) e o agrupamento em `mild_or_moderate` é defensável: ambas as classes representam demência diagnosticada, separadas apenas pela severidade. Binário fica como **fallback obrigatório** caso o desempenho em 3 classes seja insuficiente — o `Dataset` PyTorch deve expor as duas opções via flag.

## Consequências

- Desbalanceamento ainda significativo (~12× entre `non_demented` e `mild_or_moderate`). Justifica métricas balanceadas (ver ADR-0005) e técnicas de re-amostragem ou class-weighted loss no treino.
- McNemar continua viável (modelos comparados par-a-par no mesmo test set).
- Dataset/dataloader implementam flag `label_scheme` aceitando `"3class"` ou `"binary"`.
- A confirmar com orientador na primeira reunião.

## Referências

- Findings: F-0002 (contagem de pacientes)
- CLAUDE.md, seção "Decisões obrigatórias"
