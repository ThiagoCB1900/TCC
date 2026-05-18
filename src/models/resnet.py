"""ResNet-50 com pesos ImageNet via timm (baseline obrigatório do TCC).

Wrapper fino para padronizar interface entre modelos (ResNet-50, ViT-Base, Swin-T).
Cada `build_*` retorna `(model, info_dict)` para serialização da config (ADR-0008).
"""
from __future__ import annotations

import timm
import torch


def build_resnet50(
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> tuple[torch.nn.Module, dict]:
    """ResNet-50 (He et al. 2016), pesos ImageNet via timm.

    A classification head é re-inicializada com `num_classes` outputs (fine-tune).
    Backbone fica trainable por default (full fine-tune); usar `freeze_backbone=True`
    via param adicional se quisermos linear-probing como ablação.

    Args:
        num_classes: número de saídas do classifier.
        pretrained: usa pesos ImageNet do timm.
        drop_rate: dropout aplicado antes do classifier head (ADR-0010; default 0
            para retrocompat com V1).

    Returns:
        (model, info) — info contém 'name', 'num_params', 'pretrained', 'drop_rate',
        'head_in_features'.
    """
    model = timm.create_model(
        "resnet50",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {
        "name": "resnet50",
        "source": "timm",
        "pretrained": pretrained,
        "drop_rate": drop_rate,
        "num_classes": num_classes,
        "num_params": int(num_params),
        "head_in_features": int(model.get_classifier().in_features),
    }
    return model, info


# Para a fase 2 (ViTs), basta adicionar build_vit_base_16 e build_swin_tiny aqui,
# mantendo o mesmo contrato (model, info). O loop de treino é agnóstico.
