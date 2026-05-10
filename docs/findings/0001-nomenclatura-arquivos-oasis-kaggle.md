# F-0001 — Nomenclatura `OAS1_XXXX_MR{N}_mpr-{N}_{NNN}.jpg`

- **Data:** 2026-05-10
- **Status:** Confirmed
- **Categoria:** dataset

## O que descobrimos

Cada arquivo do dataset Kaggle (`ninadaithal/imagesoasis`) tem o padrão `OAS1_XXXX_MR{N}_mpr-{N}_{NNN}.jpg`, com a seguinte semântica:

| Componente | Significado | Valores observados |
|---|---|---|
| `OAS1` | OASIS-1 (cross-sectional) | sempre |
| `XXXX` | Subject ID, 4 dígitos zero-padded | 0001 a 0382 (com 35 buracos) |
| `MR{N}` | Sessão de RM | 1 (todos) ou 2 (apenas 19 controles, ver F-0004) |
| `mpr-{N}` | N-ésima aquisição MPRAGE T1 dentro da sessão | 1-4 normal; 5-6 só em `OAS1_0254` |
| `{NNN}` | Índice da fatia axial no volume | 100-160 (61 fatias centrais, idênticas entre classes) |

`mpr-N` representa **replicatas da mesma anatomia** (3-4 aquisições T1 por sessão para reduzir ruído por média), confirmado pela baixa variação entre mpr's diferentes do mesmo `(subject, session)` (ver F-0003 para detalhe da MAE).

## Evidência

- Script: `src/data/eda.py` parseia todos os 86.437 arquivos em `Data/` com regex `^OAS1_(?P<subject>\d{4})_MR(?P<session>\d+)_mpr-(?P<mpr>\d+)_(?P<slice_idx>\d+)\.jpg$`. Zero arquivos rejeitados.
- Manifesto: `results/eda/manifest.csv` (86.437 linhas).
- Sumário JSON: `results/eda/summary.json` campos `unique_sessions`, `mpr_unique`, `slice_idx_range_global`.
- Confirmação semântica via Marcus et al. 2007 (paper original OASIS) e descrição do Kaggle.

## Implicação

- Subject ID é a chave de agrupamento para o split por paciente (ver ADR-0002).
- `(subject, session, mpr)` define uma aquisição única (1.417 totais).
- Slices 100-160 não introduzem viés de profundidade entre classes — todas as classes cobrem a mesma faixa axial.

## Notas / armadilhas

- Inicialmente assumi que `MR1` era a única sessão. **Errado** — 19 controles têm MR1+MR2 (F-0004). Sempre auditar contra os dados reais antes de assumir uniformidade.
- O grouping inicial em `n_acquisitions` agrupava por `(subject, mpr)` ignorando session. Para os 19 controles com MR2 isso sub-contava aquisições — corrigido em commit posterior.
