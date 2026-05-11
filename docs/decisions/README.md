# Decisões metodológicas (ADR — Architecture Decision Records)

Cada arquivo `NNNN-titulo.md` registra uma decisão importante do TCC com contexto, alternativas consideradas, decisão tomada e consequências esperadas. Quando uma decisão é revertida ou substituída, **não apague** o ADR original — adicione um novo (com status `Substitui ADR-XXXX`) e marque o antigo como `Superseded`.

## Como criar um novo ADR

1. Próximo número sequencial (`0006`, `0007`, ...).
2. Nome curto descritivo: `0006-class-weights-vs-resampling.md`.
3. Use o template abaixo.
4. Adicione 1 linha no índice deste README.

## Template

```markdown
# ADR-NNNN — Título curto

- **Data:** AAAA-MM-DD
- **Status:** Proposed | Accepted | Superseded by ADR-XXXX
- **Decisores:** Thiago, [orientador?], Claude

## Contexto

Por que essa decisão precisa ser tomada agora? O que está em jogo?

## Alternativas consideradas

- **Opção A** — descrição. Prós / contras.
- **Opção B** — descrição. Prós / contras.
- **Opção C** — descrição. Prós / contras.

## Decisão

Optamos por **X**. Justificativa em 2-4 linhas, ancorada em literatura, dados ou restrições do projeto.

## Consequências

- O que isso permite ou bloqueia daqui pra frente.
- Quais ADRs/findings ficam dependentes.
- Quais riscos a decisão introduz e como mitigá-los.

## Referências

- Findings que sustentam: F-NNNN, ...
- Papers: ...
- Conversas/reuniões: ...
```

## Índice

| ADR | Título | Status | Data |
|---|---|---|---|
| [0001](0001-esquema-de-classes-3-classes.md) | Esquema de classes: 3 classes (Mild+Moderate fundidos) | Accepted | 2026-05-10 |
| [0002](0002-split-estratificado-por-paciente.md) | Split estratificado por subject ID, nunca por slice | Accepted | 2026-05-10 |
| [0003](0003-hardware-colab-para-treino.md) | Treino no Colab, dev local em CPU (RX6600 não viável) | Accepted | 2026-05-10 |
| [0004](0004-preprocessamento-resize-224.md) | Resize 224×224 squash, manter RGB sintético | Accepted | 2026-05-10 |
| [0005](0005-metricas-primarias.md) | Métricas primárias: macro-F1, balanced accuracy, AUC | Accepted | 2026-05-10 |
| [0006](0006-dataset-augmentations-e-label-encoding.md) | Dataset PyTorch: augmentations leves e label encoding por severidade | Accepted | 2026-05-10 |
| [0007](0007-tratamento-de-desbalanceamento.md) | Weighted CrossEntropy "balanced" com ablação sem peso | Accepted | 2026-05-10 |
| [0008](0008-framework-treino-pytorch-puro.md) | Framework de treino: PyTorch puro com torchmetrics + YAML (sem Lightning) | Accepted | 2026-05-10 |
