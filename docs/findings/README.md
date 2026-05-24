# Findings — descobertas factuais e auditadas

Cada arquivo `NNNN-titulo.md` registra uma descoberta concreta sobre o dataset, o domínio, ou trabalhos relacionados, com a evidência que a sustenta e a implicação para o TCC. **Findings descrevem o que é**, não o que decidimos fazer (isso são ADRs).

## Como criar um novo finding

1. Próximo número sequencial (`0006`, `0007`, ...).
2. Nome curto: `0006-distribuicao-cdr-por-idade.md`.
3. Use o template abaixo.
4. Adicione 1 linha no índice deste README.
5. Se um finding sustenta uma decisão, **referencie-o no ADR correspondente**.

## Template

```markdown
# F-NNNN — Título curto

- **Data:** AAAA-MM-DD
- **Status:** Confirmed | Tentative | Superseded by F-XXXX
- **Categoria:** dataset | imagem | metodologia | trabalho-relacionado | clinico

## O que descobrimos

Resumo em 1-3 frases.

## Evidência

- Script/comando que gerou (caminho relativo + parâmetros).
- Saída numérica relevante (tabela, valor agregado).
- Figura(s) salva(s), se houver: `results/.../figura.png`.

## Implicação

O que isso muda no projeto. Liga para ADRs ou outros findings.

## Notas / armadilhas

Qualquer coisa que tenha confundido durante a investigação. Importante para não repetir.
```

## Índice

| Finding | Título | Status | Data |
|---|---|---|---|
| [0001](0001-nomenclatura-arquivos-oasis-kaggle.md) | Nomenclatura `OAS1_XXXX_MR{N}_mpr-{N}_{NNN}.jpg` | Confirmed | 2026-05-10 |
| [0002](0002-discrepancia-numero-pacientes.md) | Discrepância de pacientes: 461 anunciado / 416 oficial / 347 reais | Confirmed | 2026-05-10 |
| [0003](0003-imagens-sao-slices-axiais-unicos.md) | Imagens 496×248 são slices axiais únicos (não dual-view) | Confirmed | 2026-05-10 |
| [0004](0004-mr2-reliability-subset.md) | MR2 em 19 controles é o sub-conjunto reliability do OASIS-1 | Confirmed | 2026-05-10 |
| [0005](0005-notebook-cnn-falhas-metodologicas.md) | Notebook CNN comparativo (uraninjo dataset) — 5 falhas metodológicas | Confirmed | 2026-05-10 |
| [0006](0006-gitignore-windows-case-insensitive.md) | `.gitignore` `Data/` sem âncora ignora `src/data/` no Windows | Confirmed | 2026-05-10 |
| [0007](0007-val-test-batches-agrupados-por-classe.md) | Batches de val/test sem shuffle ficam agrupados por classe (não é bug) | Confirmed | 2026-05-10 |
| [0008](0008-rancom-vit-mesmo-dataset-metodologia-comprometida.md) | RanCom-ViT (Lu 2025) usa o mesmo dataset com metodologia comprometida | Confirmed | 2026-05-10 |
| [0009](0009-survey-tratamento-desbalanceamento-na-literatura.md) | Como a literatura de ViT+Alzheimer trata desbalanceamento (varredura sistemática) | Confirmed | 2026-05-10 |
| [0010](0010-smoke-test-eval-truncado-bug-corrigido.md) | Smoke test com eval truncado escondia ausência de classes; corrigido com `shuffle_eval` | Confirmed | 2026-05-10 |
| [0011](0011-drive-symlink-i-o-bottleneck-no-colab.md) | Ler `Data/` direto do Drive (symlink) é gargalo severo de I/O no Colab | Confirmed | 2026-05-10 |
| [0012](0012-kaggle-notebooks-resolve-i-o-do-drive.md) | Kaggle Notebooks resolve nativamente o gargalo de I/O do F-0011 | Confirmed | 2026-05-10 |
| [0013](0013-baseline-v1-overfit-rapido-weighted-validado.md) | Baseline V1: weighted CE validado, mas overfit catastrófico em 4 epochs | Confirmed | 2026-05-17 |
| [0014](0014-auditoria-mestre-split-por-paciente-nos-trabalhos.md) | Auditoria-mestre: split por paciente nos trabalhos relacionados (alicerce do TCC) | Confirmed | 2026-05-17 |
| [0015](0015-mapa-de-posicionamento-e-gap-na-literatura.md) | Mapa de posicionamento: métricas, oponente direto (RanCom-ViT) e o gap (ViT+OASIS+split por paciente) | Confirmed | 2026-05-17 |
| [0016](0016-praticas-de-preprocessamento-balanceamento-augmentation.md) | Práticas de pré-proc/balanceamento/augmentation (papers + best-practices); balanceamento só no treino | Confirmed | 2026-05-17 |
| [0017](0017-baseline-v2-overfit-controlado-teto-resnet.md) | Baseline V2: overfit controlado, mas teto do ResNet-50 em ~0,59-0,62 balanced_acc | Confirmed | 2026-05-22 |
| [0018](0018-transformers-overfit-rapido-swin-melhor.md) | Transformers overfittam rápido com hiperparâmetros V2; Swin-T é o melhor modelo (bal_acc 0,594) | Confirmed | 2026-05-23 |
| [0019](0019-v3-transformer-destravou-swin-melhor-modelo.md) | V3-transformer destravou ganho; Swin-T V3 é o melhor modelo (bal_acc 0,616, AUC 0,836) | Confirmed | 2026-05-23 |
