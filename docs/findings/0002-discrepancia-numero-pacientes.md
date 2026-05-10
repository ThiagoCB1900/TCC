# F-0002 — Discrepância de pacientes: 461 anunciado / 416 oficial / 347 reais

- **Data:** 2026-05-10
- **Status:** Confirmed
- **Categoria:** dataset

## O que descobrimos

Três números diferentes coexistem para a contagem de pacientes do OASIS-1:

| Fonte | Pacientes | Confiabilidade |
|---|---:|---|
| Texto do Kaggle (`ninadaithal/imagesoasis`) | 461 | **Provavelmente errado** |
| OASIS-1 oficial (Marcus et al. 2007) | **416** | Canônico |
| Encontrado em `Data/` (este projeto) | **347** | Auditado |

A versão Kaggle disponibilizada **tem 347 pacientes únicos** (~83% do OASIS-1 oficial), com IDs de OAS1_0001 a OAS1_0382 e 35 buracos internos (ver lista completa em `results/eda/manifest.csv` ou no log do script de auditoria). Pacientes com IDs ≥ 383 estão totalmente ausentes.

Distribuição por classe (3 classes, ver ADR-0001):

| Classe | Pacientes | Aquisições | Slices | % slices |
|---|---:|---:|---:|---:|
| non_demented | 266 | 1.102 | 67.222 | 77,8% |
| very_mild | 58 | 225 | 13.725 | 15,9% |
| mild_or_moderate | 23 | 90 | 5.490 | 6,4% |
| **Total** | **347** | **1.417** | **86.437** | 100% |

## Evidência

- `results/eda/summary.json` campos `n_subjects`, `n_acquisitions`, `n_slices`.
- `results/eda/eda_report.md` tabelas por esquema de classes.
- Auditoria de IDs: 347 únicos, range 1-382, 35 ausentes na faixa contígua.
- Slices/paciente: mediana 244 (= 4 mpr × 61), max 488 (8 acq, controles MR1+MR2), min 122 (2 mpr × 61).

## Implicação

- **Não é download incompleto** — é o que o Kaggle disponibilizou. Não vale a pena perseguir os ~69 pacientes faltantes vs OASIS-1 oficial agora; o aluno teria que baixar do site oficial OASIS e re-processar.
- O TCC deve **mencionar essa limitação** na seção de dataset/metodologia: "trabalhamos com 347 dos 416 pacientes do OASIS-1 oficial, conforme distribuído pela versão Kaggle pré-processada utilizada".
- O número "461 patients" anunciado pelo uploader Kaggle é provavelmente incorreto (talvez tenha contado replicatas mpr ou somou disc parts). **Não citar o 461 no TCC** — citar 347 (real) e 416 (canônico OASIS-1).
- Desbalanceamento é severo (~138× em slices entre maior e menor classe; 2 pacientes em Moderate). Justifica diretamente ADR-0001 (3 classes), ADR-0002 (split estratificado), ADR-0005 (métricas balanceadas).

## Notas / armadilhas

- Dataset publicamente disponível como "OASIS" pode não bater com OASIS oficial. Sempre auditar.
- A conversão de "número de imagens anunciado" para "número de pacientes" não é trivial: cada paciente contribui 244 slices (em média), então "80.000 imagens" do anúncio Kaggle deveria implicar ~328 pacientes — mais próximo do real (347) do que do anunciado (461).
