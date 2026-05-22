"""ViT-Base/16 com pesos ImageNet via timm.

Mesmo contrato dos outros builders: retorna (model, info). O loop de treino
(src/training/loop.py) e o factory (src/models/factory.py) são agnósticos à
arquitetura — a única mudança vs ResNet é a construção do modelo.
"""
from __future__ import annotations

import timm
import torch


def build_vit_base16(
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> tuple[torch.nn.Module, dict]:
    """ViT-Base/16 (Dosovitskiy et al. 2020), pesos ImageNet via timm.

    Args:
        num_classes: saídas do classifier head.
        pretrained: usa pesos pré-treinados (augreg in21k → in1k, default timm).
        drop_rate: dropout no classifier head.
        drop_path_rate: stochastic depth no backbone (regularização típica de ViT;
            ADR-0010 menciona drop_path como técnica para transformers).

    Entrada esperada: 224×224 (compatível com nosso image_size e ADR-0004).
    """
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {
        "name": "vit_base_16",
        "timm_id": "vit_base_patch16_224",
        "source": "timm",
        "pretrained": pretrained,
        "drop_rate": drop_rate,
        "drop_path_rate": drop_path_rate,
        "num_classes": num_classes,
        "num_params": int(num_params),
        "head_in_features": int(model.get_classifier().in_features),
    }
    return model, info
