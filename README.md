# Vision Transformers para Classificação de Alzheimer

Trabalho de Conclusão de Curso (TCC) · Ciências da Computação · UECE.

Comparação entre **Vision Transformers** (ViT-Base/16 e Swin-T) e um baseline CNN (ResNet-50) na classificação de imagens de ressonância magnética cerebral do dataset **OASIS-1**, com análise de interpretabilidade (Attention Rollout e Grad-CAM).

## Estrutura

```
TCC/
├── data/                  # OASIS bruto (não versionado)
├── notebooks/             # EDA e prototipação
├── src/
│   ├── data/              # preprocessing, splits, dataloaders
│   ├── models/            # ViT, Swin, ResNet
│   ├── training/          # loops de treino
│   ├── interpretability/  # attention rollout, grad-cam
│   └── evaluation/        # métricas, McNemar
├── experiments/           # configs e logs por run
├── results/               # tabelas, figuras, attention maps
├── docs/                  # papers lidos, fichamentos, draft do TCC
├── CLAUDE.md              # contexto do projeto para Claude Code
└── README.md
```

## Stack

Python · PyTorch · timm · MONAI

## Princípio metodológico crítico

**Split por paciente (subject ID), nunca por slice.** Slices do mesmo paciente em conjuntos diferentes inflam acurácia e invalidam o trabalho.

## Status

Em desenvolvimento — plano de 60 dias iniciado em 2026-05-07. Ver [CLAUDE.md](CLAUDE.md) para o estado atual detalhado.
