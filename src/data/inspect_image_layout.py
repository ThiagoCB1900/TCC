"""Inspeciona propriedades das imagens 2D do dataset Kaggle.

Cada arquivo .jpg contém UM único slice axial alongado horizontalmente
(496x248, aspect ratio ~2:1 vindo da conversão NIfTI->JPG). Este script
checa apenas o que de fato precisa ser verificado:

- Os 3 canais RGB são iguais (grayscale promovido)?
- O contorno do cérebro está bem centralizado, ou há offset que comprometa
  o resize para 224x224?

Salva uma figura com 8 amostras (2 por classe) em
results/eda/figures/sample_grid.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DATA_ROOT = Path("Data")
OUT_FIG = Path("results/eda/figures/sample_grid.png")


def main() -> int:
    df = pd.read_csv("results/eda/manifest.csv")
    rng = np.random.default_rng(seed=11)

    # 2 amostras por classe, slice central (130) quando disponível
    samples = []
    for cls in ["non_demented", "very_mild", "mild", "moderate"]:
        sub = df[(df["class_4"] == cls) & (df["slice_idx"] == 130)]
        if len(sub) < 2:
            sub = df[df["class_4"] == cls]
        idx = rng.choice(len(sub), size=2, replace=False)
        samples.extend(sub.iloc[idx].to_dict(orient="records"))

    # Verifica igualdade entre canais RGB
    print("=== Igualdade entre canais RGB (R == G == B?) ===")
    rgb_eq_count = 0
    aspect_ratios = []
    for s in samples:
        full = DATA_ROOT.parent / s["path"]
        with Image.open(full) as img:
            arr = np.asarray(img)
        all_eq = (
            np.array_equal(arr[..., 0], arr[..., 1])
            and np.array_equal(arr[..., 1], arr[..., 2])
        )
        rgb_eq_count += int(all_eq)
        h, w = arr.shape[:2]
        aspect_ratios.append(w / h)
        print(f"  {s['class_4']:13s} {s['subject']} slice {s['slice_idx']}: R==G==B? {all_eq} | {w}x{h}")
    print(f"\n  {rgb_eq_count}/{len(samples)} imagens com R==G==B (grayscale promovido).")
    print(f"  Aspect ratio (largura/altura): media={np.mean(aspect_ratios):.3f}")

    # Figura: 8 amostras
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()
    for ax, s in zip(axes, samples):
        full = DATA_ROOT.parent / s["path"]
        with Image.open(full) as img:
            ax.imshow(img, cmap="gray")
        ax.set_title(f"{s['class_4']}\n{s['subject']} slice {s['slice_idx']}", fontsize=9)
        ax.axis("off")
    plt.suptitle("Amostras: 2 por classe, slice central (~130)", fontsize=12)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=130)
    plt.close(fig)
    print(f"\n  Figura salva em {OUT_FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
