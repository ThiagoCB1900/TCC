"""Grad-CAM (Selvaraju et al., 2017) — explicação por gradientes para CNN e transformer.

Grad-CAM pondera as ativações de uma camada-alvo pelos gradientes da classe de
interesse, destacando as regiões que mais influenciam a predição.

- ResNet-50 (timm): camada-alvo `model.layer4` — ativações já em formato NCHW.
- Swin-T (timm): tokens precisam de `reshape_transform` para virar grade espacial.

Para ViT preferimos Attention Rollout (attention_rollout.py); Grad-CAM em ViT é
possível mas menos natural. Aqui cobrimos CNN (ResNet) e Swin.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


class GradCAM:
    """Grad-CAM para uma camada-alvo. Use como context manager."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        reshape_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        """
        Args:
            model: modelo treinado (eval).
            target_layer: módulo cujas ativações/gradientes serão usados
                (ex: ResNet `model.layer4`; Swin último estágio).
            reshape_transform: para transformers, converte a ativação capturada
                ([B,H,W,C] ou [B,L,C]) em [B,C,h,w]. None para CNN (já é NCHW).
        """
        self.model = model
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._hooks: list = []

    def _fwd_hook(self, _m, _i, out):
        self._activations = out

    def _bwd_hook(self, _m, _gi, gout):
        self._gradients = gout[0]

    def __enter__(self) -> "GradCAM":
        self._hooks.append(self.target_layer.register_forward_hook(self._fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(self._bwd_hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """Gera o CAM para x (1,3,H,W). Se class_idx None, usa a classe predita.

        Returns: (cam [h,w] em [0,1], class_idx usado).
        """
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError("GradCAM espera batch de 1 imagem (1,3,H,W).")
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=-1).item())
        score = logits[0, class_idx]
        score.backward()

        acts = self._activations
        grads = self._gradients
        if acts is None or grads is None:
            raise RuntimeError("Ativações/gradientes não capturados; target_layer correto?")

        if self.reshape_transform is not None:
            acts = self.reshape_transform(acts)
            grads = self.reshape_transform(grads)

        # acts/grads: [B, C, h, w]
        weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP dos gradientes
        cam = (weights * acts).sum(dim=1).squeeze(0)     # [h, w]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), class_idx


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Converte ativação do Swin (timm) para [B, C, h, w].

    timm Swin produz, no último estágio, [B, H, W, C] (NHWC). Alguns formatos
    vêm como [B, L, C] (tokens). Detecta e normaliza para [B, C, h, w].
    """
    if tensor.dim() == 4:
        # [B, H, W, C] -> [B, C, H, W]
        return tensor.permute(0, 3, 1, 2).contiguous()
    if tensor.dim() == 3:
        # [B, L, C] -> [B, C, h, w] (assume grade quadrada)
        b, l, c = tensor.shape
        h = int(l ** 0.5)
        return tensor.reshape(b, h, h, c).permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"formato de ativacao Swin inesperado: {tuple(tensor.shape)}")
