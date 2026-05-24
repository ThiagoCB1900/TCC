"""Attention Rollout para ViT (Abnar & Zuidema, 2020).

Ideia: a atenção de uma camada não conta a história toda — a informação flui
através das camadas. O rollout multiplica as matrizes de atenção de todas as
camadas (somando a conexão residual via identidade) para estimar quanto cada
patch de entrada contribui para o token de classificação (CLS).

Implementação para o ViT do timm:
- timm usa `fused_attn` (scaled_dot_product_attention) por padrão, que NÃO expõe
  a matriz de atenção. Forçamos `fused_attn=False` e capturamos a matriz via
  forward-hook no `attn_drop` (cujo input é exatamente a matriz pós-softmax,
  shape [B, heads, N, N]).
- Rollout: A_hat_l = 0.5*A_l + 0.5*I (residual); produto acumulado L..1;
  linha do CLS (índice 0) → patches → grid 14×14 → upsample 224×224.

Referências: attention_flow_in_transformers.pdf (Abnar), Chefer CVPR 2021 (alternativa).
"""
from __future__ import annotations

import numpy as np
import torch


class AttentionRollout:
    """Captura matrizes de atenção do ViT e computa o rollout para uma imagem."""

    def __init__(self, model: torch.nn.Module, head_fusion: str = "mean", discard_ratio: float = 0.0):
        """
        Args:
            model: ViT do timm (vit_base_patch16_224).
            head_fusion: como fundir as cabeças de atenção ('mean' | 'max' | 'min').
            discard_ratio: fração das menores atenções a zerar por camada (reduz ruído;
                0 = nenhum descarte). Abnar usa rollout puro (0); Chefer/derivados às
                vezes descartam as menores para realçar.
        """
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self._attentions: list[torch.Tensor] = []
        self._hooks: list = []

    def _hook(self, _module, inp, _out):
        # input[0] do attn_drop é a matriz de atenção pós-softmax [B, heads, N, N]
        self._attentions.append(inp[0].detach().cpu())

    def __enter__(self) -> "AttentionRollout":
        for blk in self.model.blocks:
            # força caminho de atenção explícita (necessário para o hook ver a matriz)
            if hasattr(blk.attn, "fused_attn"):
                blk.attn.fused_attn = False
            self._hooks.append(blk.attn.attn_drop.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> np.ndarray:
        """Roda o modelo em x (1,3,H,W) e retorna o mapa de rollout (H',W') em [0,1].

        H'×W' é o grid de patches (14×14 para ViT-Base/16 em 224) — faça upsample
        para o tamanho da imagem ao sobrepor.
        """
        if x.dim() != 4 or x.shape[0] != 1:
            raise ValueError("AttentionRollout espera batch de 1 imagem (1,3,H,W).")
        self._attentions.clear()
        self.model.eval()
        _ = self.model(x)

        if not self._attentions:
            raise RuntimeError(
                "Nenhuma matriz de atenção capturada. O modelo é um ViT do timm com blocks[].attn.attn_drop?"
            )

        # Fusão das cabeças por camada → lista de [N, N]
        result = torch.eye(self._attentions[0].size(-1))
        for attn in self._attentions:
            a = attn[0]  # [heads, N, N]
            if self.head_fusion == "mean":
                a = a.mean(dim=0)
            elif self.head_fusion == "max":
                a = a.max(dim=0).values
            elif self.head_fusion == "min":
                a = a.min(dim=0).values
            else:
                raise ValueError(self.head_fusion)

            if self.discard_ratio > 0:
                flat = a.view(-1)
                k = int(flat.numel() * self.discard_ratio)
                if k > 0:
                    _, idx = flat.topk(k, largest=False)
                    # não descartar a coluna/linha do CLS (índice 0)
                    idx = idx[idx != 0]
                    flat[idx] = 0
                    a = flat.view_as(a)

            # residual + renormalização
            a = a + torch.eye(a.size(-1))
            a = a / a.sum(dim=-1, keepdim=True)
            result = a @ result

        # Linha do CLS (0) → atenção sobre os patches (descarta o próprio CLS)
        mask = result[0, 1:]
        n_patches = mask.numel()
        grid = int(n_patches ** 0.5)
        mask = mask.reshape(grid, grid).numpy()
        mask = mask / (mask.max() + 1e-8)
        return mask


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Sobrepõe um heatmap (H',W' em [0,1]) sobre a imagem RGB (H,W,3 uint8).

    Faz upsample do heatmap para o tamanho da imagem com interpolação bilinear.
    Retorna uint8 (H,W,3) para salvar/plotar.
    """
    import cv2

    h, w = image_rgb.shape[:2]
    hm = cv2.resize((heatmap * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    out = (alpha * hm_color + (1 - alpha) * image_rgb).clip(0, 255).astype(np.uint8)
    return out
