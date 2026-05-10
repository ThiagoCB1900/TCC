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
