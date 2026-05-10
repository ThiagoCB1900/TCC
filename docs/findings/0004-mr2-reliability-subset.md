# F-0004 — MR2 em 19 controles é o sub-conjunto reliability do OASIS-1

- **Data:** 2026-05-10
- **Status:** Confirmed
- **Categoria:** dataset

## O que descobrimos

19 sujeitos do dataset têm sessões `MR1` **e** `MR2` (mesmo subject ID, duas sessões de RM). **Todos 19 são `non_demented`**. Não há leak de classe entre as sessões.

Esses 19 são o sub-conjunto de reliability do OASIS-1 original: Marcus et al. 2007 reescanearam ~20 controles em sessões separadas para avaliar reprodutibilidade do protocolo. **Não é OASIS-2 (longitudinal)** — OASIS-1 é cross-sectional, e o reliability subset é uma exceção planejada para validação técnica.

## Evidência

- Auditoria via pandas (script removido após investigação, dados em `results/eda/manifest.csv`).
- IDs com MR2: OAS1_0061, 0080, 0092, 0101, 0111, 0117, 0145, 0150, 0156, 0191, 0202, 0230, 0236, 0239, 0249, 0285, 0353, 0368, 0379.
- Todos com `class_4 == "non_demented"`.
- Distribuição de slices por paciente: pacientes com MR2 têm 488 slices (= 8 acq × 61), o dobro dos demais (244).
- mpr também foi auditado: vai até 6 em **um** sujeito (`OAS1_0254`), provavelmente outra exceção do protocolo. Outlier benigno.

## Implicação

- Para o **split por paciente** (ADR-0002), basta agrupar por `subject` — automaticamente coloca MR1 e MR2 do mesmo sujeito no mesmo split. Sem ação extra.
- Para análise de **estabilidade**, esses 19 com MR2 podem ser usados em uma análise extra opcional: mostrar que predições do modelo são consistentes entre sessões do mesmo sujeito (test-retest reliability). Cabe na semana 6 se houver tempo. **Não é prioritário.**
- Para `n_acquisitions` no manifesto, agrupar por `(subject, session, mpr)` — ignorar `session` sub-conta as MR2 dos 19 controles.

## Notas / armadilhas

- Inicialmente assumi "OASIS-1 só tem MR1" e a EDA não destacaria a sessão. Bug pego ao ver `unique_sessions: [1, 2]` nos sanity checks. Sempre incluir contagens de cardinalidade dos campos no sanity check.
- "Mesmo paciente em duas sessões" não é leakage clínico — é literalmente o mesmo paciente, mesma classe, scaneado duas vezes. **Mas** ainda é leakage **estatístico** se cair em train/test splits diferentes (modelo memoriza). Por isso o split por subject ID é mandatório (ADR-0002).
