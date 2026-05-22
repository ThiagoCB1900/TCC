"""Swin-T (Tiny) com pesos ImageNet via timm.

Mesmo contrato: retorna (model, info). Entrada 224×224 (variante window7_224).
"""
from __future__ import annotations

import timm
import torch


def build_swin_tiny(
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> tuple[torch.nn.Module, dict]:
    """Swin Transformer Tiny (Liu et al. 2021), pesos ImageNet via timm.

    Args:
        num_classes: saídas do classifier head.
        pretrained: usa pesos pré-treinados ImageNet-1k (default timm).
        drop_rate: dropout no classifier head.
        drop_path_rate: stochastic depth no backbone.

    Entrada esperada: 224×224 (variante swin_tiny_patch4_window7_224).
    """
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {
        "name": "swin_tiny",
        "timm_id": "swin_tiny_patch4_window7_224",
        "source": "timm",
        "pretrained": pretrained,
        "drop_rate": drop_rate,
        "drop_path_rate": drop_path_rate,
        "num_classes": num_classes,
        "num_params": int(num_params),
        "head_in_features": int(model.get_classifier().in_features),
    }
    return model, info
