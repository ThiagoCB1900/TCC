"""ResNet (50 e 18) com pesos ImageNet via timm.

Wrapper fino para padronizar interface entre modelos (ResNet, ViT-Base, Swin-T).
Cada `build_*` retorna `(model, info_dict)` para serialização da config (ADR-0008).

ResNet-50 = baseline obrigatório (ADR-0008). ResNet-18 = ablação de capacidade
(F-0021 / hipótese "menos parâmetros generaliza melhor com poucos dados"),
mantendo transfer learning.
"""
from __future__ import annotations

import timm
import torch

# nome curto do projeto -> timm id
_RESNET_TIMM_IDS = {
    "resnet50": "resnet50",
    "resnet18": "resnet18",
}


def _build_resnet(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    drop_rate: float = 0.0,
) -> tuple[torch.nn.Module, dict]:
    """Constrói uma variante de ResNet do timm com classifier re-inicializado.

    A classification head é re-inicializada com `num_classes` outputs (fine-tune).
    Backbone trainable por default (full fine-tune).
    """
    timm_id = _RESNET_TIMM_IDS[name]
    model = timm.create_model(
        timm_id,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = {
        "name": name,
        "timm_id": timm_id,
        "source": "timm",
        "pretrained": pretrained,
        "drop_rate": drop_rate,
        "num_classes": num_classes,
        "num_params": int(num_params),
        "head_in_features": int(model.get_classifier().in_features),
    }
    return model, info


def build_resnet50(num_classes: int, pretrained: bool = True, drop_rate: float = 0.0):
    """ResNet-50 (He et al. 2016), pesos ImageNet via timm (~23,5M params)."""
    return _build_resnet("resnet50", num_classes, pretrained, drop_rate)


def build_resnet18(num_classes: int, pretrained: bool = True, drop_rate: float = 0.0):
    """ResNet-18 (He et al. 2016), pesos ImageNet via timm (~11,2M params).

    Ablação de capacidade: ~metade dos params da ResNet-50, mantendo transfer
    learning — testa se 'menos parâmetros generaliza melhor' no nosso regime de
    dados pequenos (242 pacientes).
    """
    return _build_resnet("resnet18", num_classes, pretrained, drop_rate)
