"""Gera mapas de interpretabilidade a partir de um checkpoint treinado.

- ViT  -> Attention Rollout (attention_rollout.py)
- ResNet/Swin -> Grad-CAM (gradcam.py)

Roda LOCAL em CPU (leve: forward de poucas imagens). Precisa apenas de:
- checkpoint_best.pt da run (baixar do Kaggle para experiments/runs/<run>/)
- Data/ local + manifest + split (já no repo)

Uso:
    python -m src.interpretability.generate_maps \
        --run-dir experiments/runs/20260523_181850_swin_tiny_v3_class3_weighted_kaggle \
        --n-per-class 3

Saída: results/interpretability/<run>/grid.png + overlays individuais.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    OASISDataset,
)
from src.interpretability.attention_rollout import AttentionRollout, overlay_heatmap
from src.interpretability.gradcam import GradCAM, swin_reshape_transform
from src.models.factory import build_model


def load_model_from_checkpoint(ckpt_path: Path, num_classes: int, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    info = ckpt.get("model_info", {})
    name = info.get("name") or cfg.get("model_name")
    if name is None:
        raise ValueError("Nao consegui inferir a arquitetura do checkpoint (config/model_info ausentes).")
    model, _ = build_model(name, num_classes=num_classes, pretrained=False,
                           drop_rate=cfg.get("drop_rate", 0.0),
                           drop_path_rate=cfg.get("drop_path_rate", 0.0))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, name


def make_map(model, name: str, x: torch.Tensor) -> tuple[np.ndarray, int | None]:
    """Retorna (heatmap [h,w] em [0,1], class_idx ou None)."""
    if name == "vit_base_16":
        with AttentionRollout(model, head_fusion="mean") as ar:
            hm = ar(x)
        return hm, None
    if name == "resnet50":
        with GradCAM(model, model.layer4) as cam:
            return cam(x)
    if name == "swin_tiny":
        with GradCAM(model, model.layers[-1], reshape_transform=swin_reshape_transform) as cam:
            return cam(x)
    raise ValueError(f"metodo de interpretabilidade nao definido para {name}")


def denorm_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True, help="Pasta da run com checkpoint_best.pt")
    p.add_argument("--manifest", type=Path, default=Path("results/eda/manifest.csv"))
    p.add_argument("--split", type=Path, default=Path("experiments/splits/split_v1.json"))
    p.add_argument("--data-root", type=Path, default=Path("."))
    p.add_argument("--label-scheme", default="class_3")
    p.add_argument("--n-per-class", type=int, default=3)
    p.add_argument("--out-dir", type=Path, default=Path("results/interpretability"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    ckpt = args.run_dir / "checkpoint_best.pt"
    if not ckpt.is_file():
        print(f"[ERRO] checkpoint nao encontrado: {ckpt}")
        print("Baixe o checkpoint_best.pt da run no Kaggle para essa pasta.")
        return 2

    device = torch.device("cpu")
    n_classes = 3 if args.label_scheme == "class_3" else 2
    model, name = load_model_from_checkpoint(ckpt, n_classes, device)
    print(f"Modelo: {name} | metodo: {'Attention Rollout' if name=='vit_base_16' else 'Grad-CAM'}")

    ds = OASISDataset(
        manifest_path=args.manifest, split_path=args.split, fold="test",
        label_scheme=args.label_scheme, data_root=args.data_root, augment_strength="off",
    )
    idx_to_class = ds.idx_to_class

    # seleciona n amostras por classe
    rng = np.random.default_rng(args.seed)
    chosen: list[int] = []
    for cls_idx in range(n_classes):
        pool = np.where(ds._labels == cls_idx)[0]
        if len(pool):
            chosen.extend(rng.choice(pool, size=min(args.n_per_class, len(pool)), replace=False).tolist())

    out_dir = args.out_dir / args.run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    ncols = 2  # imagem original | overlay
    fig, axes = plt.subplots(len(chosen), ncols, figsize=(2 * ncols + 1, 2.2 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]

    for row, idx in enumerate(chosen):
        x, label = ds[int(idx)]
        x = x.unsqueeze(0).to(device)
        hm, pred = make_map(model, name, x)
        # predicao para o titulo
        with torch.no_grad():
            logits = model(x)
            pred_cls = int(logits.argmax(-1).item())
        orig = denorm_to_uint8(x.squeeze(0))
        over = overlay_heatmap(orig, hm, alpha=0.5)

        true_name = idx_to_class[int(label)]
        pred_name = idx_to_class[pred_cls]
        ok = "OK" if pred_cls == label else "ERRO"
        axes[row, 0].imshow(orig); axes[row, 0].axis("off")
        axes[row, 0].set_title(f"{ds._subjects[idx]}\nreal: {true_name}", fontsize=8)
        axes[row, 1].imshow(over); axes[row, 1].axis("off")
        axes[row, 1].set_title(f"pred: {pred_name} [{ok}]", fontsize=8)

        # salva overlay individual
        Image.fromarray(over).save(out_dir / f"{row:02d}_{true_name}_{ds._subjects[idx]}.png")

    method = "Attention Rollout" if name == "vit_base_16" else "Grad-CAM"
    plt.suptitle(f"{name} — {method} (test set)", fontsize=11)
    plt.tight_layout()
    grid_path = out_dir / "grid.png"
    plt.savefig(grid_path, dpi=130)
    plt.close(fig)
    print(f"OK. Mapas salvos em {out_dir} (grid.png + overlays individuais).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
