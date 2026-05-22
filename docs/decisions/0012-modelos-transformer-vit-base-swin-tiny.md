# ADR-0012 — Modelos transformer: ViT-Base/16 e Swin-T (variantes timm, mesma metodologia)

- **Data:** 2026-05-22
- **Status:** Accepted
- **Decisores:** Thiago, Claude

## Contexto

O baseline ResNet-50 está fechado (F-0017: teto ~0,59 balanced_acc; gargalo deixou de ser overfit e passou a ser representação). O objetivo central do TCC são os Vision Transformers. Precisamos fixar **quais variantes** treinar e garantir que rodem sob a **mesma metodologia** do baseline (mesmo split, loop, regras de ouro) para comparação justa.

## Decisão

Treinar **ViT-Base/16** e **Swin-T**, via timm, com pesos ImageNet, sob o pipeline definitivo (ADR-0011) e os mesmos hiperparâmetros V2 (ADR-0010), mudando **apenas a arquitetura**.

| Modelo | timm id | Entrada | Params | Papel |
|---|---|---|---|---|
| ResNet-50 | `resnet50` | 224×224 | 23,5M | baseline (fechado, F-0017) |
| **ViT-Base/16** | `vit_base_patch16_224` | 224×224 | 85,8M | transformer puro (Dosovitskiy 2020) |
| **Swin-T** | `swin_tiny_patch4_window7_224` | 224×224 | 27,5M | transformer hierárquico (Liu 2021) |

### Implementação

- Builders: `src/models/vit.py::build_vit_base16`, `src/models/swin.py::build_swin_tiny`, despachados por `src/models/factory.py::build_model`. Contrato comum `(model, info)`.
- Entrypoint: `python -m src.training.run --model {vit_base_16|swin_tiny} ...`. Defaults V2 (LR diferenciado backbone 1e-5 / head 1e-4, weight_decay 0.1, augmentation strong, patience 5, warmup 2).
- **`drop_path_rate=0.1`** (stochastic depth) ligado para ViT/Swin — regularização típica de transformers (ResNet ignora). Adicionado ao `TrainConfig` e CLI (`--drop-path-rate`).
- Mesma `image_size=224` para os três (variantes window7_224 / patch16_224). 384×384 fica como ablação opcional (Swin tem variante 384) se houver ganho a perseguir.
- `make_param_groups` usa `model.get_classifier()` do timm — funciona para ViT/Swin sem mudança.

### Comparação estatística (ADR-0005)

- O loop agora **persiste as predições por amostra do test** (`test_predictions.npz`: y_true, y_pred, y_proba, subjects) ao fim de cada run não-smoke.
- Como o test loader é determinístico (split_v1, shuffle=False), as predições ficam alinhadas por amostra entre modelos → **McNemar** par-a-par (ResNet vs ViT, ViT vs Swin, etc.) via `src/evaluation/metrics.py::mcnemar_test`.

## Alternativas consideradas

- **ViT variantes maiores (ViT-Large)**: 300M+ params, overfit garantido em 242 pacientes; fora de escopo.
- **DeiT / outras ViTs eficientes**: interessantes, mas ViT-Base é o canônico do paper-âncora (Dosovitskiy) — melhor para o TCC.
- **Swin-S/B**: maiores; Swin-T é o equivalente em escala ao ResNet-50, comparação mais justa.
- **384×384**: mais resolução, mais memória/tempo; deixar como ablação se 224 platôar.

## Consequências

- Comparação **maçã-com-maçã**: 3 arquiteturas, mesmo split/loop/métricas/regras de ouro. Diferença atribuível à arquitetura.
- **Custo de memória**: ViT-Base (85,8M) é o maior; em T4 16GB, batch 32 a 224×224 deve caber. Se OOM, reduzir para `--batch-size 16` (documentado no notebook).
- **Plano de treino** (cada ~2-3h no T4): ViT-Base weighted + Swin-T weighted como principais; ablações sem peso opcionais. Cabe na cota Kaggle (30h/semana).
- **Expectativa honesta**: ViT/Swin podem superar o ResNet (~0,59) por atenção global, ou platôar similar — neste caso a alavanca vira dados (2.5D, ADR-0011). Ambos os resultados são publicáveis no TCC.
- Interpretabilidade: ViT/Swin habilitam Attention Rollout (semana 4), diferencial sobre Grad-CAM-only do ResNet.

## Referências

- F-0017 (baseline ResNet fechado), F-0015 (gap ViT+OASIS+split por paciente).
- ADR-0010 (hiperparâmetros V2), ADR-0011 (pipeline de dados), ADR-0005 (métricas + McNemar).
- Dosovitskiy et al. 2020 (ViT); Liu et al. 2021 (Swin) — PDFs em `docs/Papers/`.
- timm: https://github.com/huggingface/pytorch-image-models
