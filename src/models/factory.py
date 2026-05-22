"""Factory de modelos — despacha por nome para o builder correto.

Mantém o loop de treino e o entrypoint agnósticos à arquitetura: todos os
builders seguem o contrato `(model, info) = build(...)`.
"""
from __future__ import annotations

import torch

from src.models.resnet import build_resnet50
from src.models.swin import build_swin_tiny
from src.models.vit import build_vit_base16

# Nomes aceitos no --model / cfg.model_name
MODEL_NAMES = ("resnet50", "vit_base_16", "swin_tiny")


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> tuple[torch.nn.Module, dict]:
    """Constrói o modelo pedido.

    `drop_path_rate` só se aplica a ViT/Swin (transformers); ResNet o ignora.
    """
    if name == "resnet50":
        return build_resnet50(num_classes, pretrained=pretrained, drop_rate=drop_rate)
    if name == "vit_base_16":
        return build_vit_base16(
            num_classes, pretrained=pretrained, drop_rate=drop_rate, drop_path_rate=drop_path_rate
        )
    if name == "swin_tiny":
        return build_swin_tiny(
            num_classes, pretrained=pretrained, drop_rate=drop_rate, drop_path_rate=drop_path_rate
        )
    raise ValueError(f"modelo nao suportado: {name!r}. Use um de {MODEL_NAMES}.")
