"""Inspeção visual e numérica do `OASISDataset` — checkpoint anti-erro.

Pega N amostras de cada fold pelo dataloader (com transforms reais aplicados),
desnormaliza para visualização, e salva uma grade em
`results/eda/figures/dataset_inspection.png`.

Imprime também:
  - contagens por classe e número de pacientes por fold;
  - estatísticas de tensor (shape, dtype, range, média/desvio);
  - diferença visual entre train (com augmentation) e val (sem augmentation):
    pega a MESMA imagem em train_ds e val_ds — se train_ds aplicou flip ou
    rotação, a imagem deve aparecer diferente.

Lição registrada (F-0003): nunca confiar em pipeline sem auditar visualmente.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    OASISDataset,
)

OUT_FIG = Path("results/eda/figures/dataset_inspection.png")
OUT_FIG_AUG = Path("results/eda/figures/dataset_augmentation_check.png")


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Converte tensor (3,H,W) normalizado ImageNet de volta para uint8 plotável."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def grid_per_fold(
    datasets: dict[str, OASISDataset], n_per_fold_per_class: int = 2
) -> None:
    """Salva uma grade [folds x classes*n] mostrando amostras processadas."""
    classes_ordered = sorted(datasets["train"].class_to_idx, key=lambda c: datasets["train"].class_to_idx[c])
    n_cols = len(classes_ordered) * n_per_fold_per_class
    n_rows = len(datasets)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    rng = np.random.default_rng(seed=7)

    for r, (fold, ds) in enumerate(datasets.items()):
        col = 0
        for cls in classes_ordered:
            cls_idx = ds.class_to_idx[cls]
            mask = ds._labels == cls_idx
            indices = np.where(mask)[0]
            if len(indices) == 0:
                for _ in range(n_per_fold_per_class):
                    axes[r, col].axis("off")
                    col += 1
                continue
            chosen = rng.choice(indices, size=min(n_per_fold_per_class, len(indices)), replace=False)
            for idx in chosen:
                tensor, label = ds[int(idx)]
                img = denormalize(tensor)
                axes[r, col].imshow(img)
                axes[r, col].set_title(
                    f"{fold} | {cls}\n{ds._subjects[idx]} (idx {idx})", fontsize=8
                )
                axes[r, col].axis("off")
                col += 1

    plt.suptitle(
        "OASISDataset — amostras processadas pelo pipeline (resize 224 + ImageNet norm)",
        fontsize=11,
    )
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=130)
    plt.close(fig)


def augmentation_diff(
    train_ds: OASISDataset,
    val_ds: OASISDataset,
    n_pairs: int = 4,
    n_aug_samples: int = 3,
) -> None:
    """Para um mesmo subject presente em train (ou usa um do train mesmo),
    mostra (val sem aug, train com aug × n_aug_samples) lado a lado.

    Como train e val usam pacientes diferentes (ADR-0002), aqui apenas verificamos
    o efeito do augmentation pegando o MESMO índice do train_ds e chamando
    __getitem__ várias vezes — espera-se variação visual."""
    rng = np.random.default_rng(seed=11)
    indices = rng.choice(len(train_ds), size=n_pairs, replace=False)

    n_cols = 1 + n_aug_samples  # 1 versão base "sem aug" reconstruída + N aug
    fig, axes = plt.subplots(n_pairs, n_cols, figsize=(2.0 * n_cols, 2.2 * n_pairs))
    if n_pairs == 1:
        axes = np.array([axes])

    # Usa val_ds.transform (sem aug) sobre as mesmas imagens-fonte para coluna 0.
    for r, idx in enumerate(indices):
        rel_path = train_ds._paths[int(idx)]
        full_path = train_ds.data_root / rel_path
        from PIL import Image as PIL_Image

        with PIL_Image.open(full_path) as img:
            img = img.convert("RGB")
        # Sem aug: aplica transform do val_ds
        baseline = val_ds.transform(img)
        axes[r, 0].imshow(denormalize(baseline))
        axes[r, 0].set_title(f"sem aug\n{train_ds._subjects[int(idx)]}", fontsize=8)
        axes[r, 0].axis("off")

        for c in range(n_aug_samples):
            tensor, _ = train_ds[int(idx)]  # cada chamada aplica aug nova
            axes[r, c + 1].imshow(denormalize(tensor))
            axes[r, c + 1].set_title(f"aug #{c+1}", fontsize=8)
            axes[r, c + 1].axis("off")

    plt.suptitle(
        "Augmentation no train: 1 baseline + 3 amostras (deve haver variação visível)",
        fontsize=11,
    )
    plt.tight_layout()
    OUT_FIG_AUG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG_AUG, dpi=130)
    plt.close(fig)


def print_summary(datasets: dict[str, OASISDataset]) -> None:
    print("=== Resumo dos datasets ===")
    for fold, ds in datasets.items():
        print(
            f"\n[{fold}] {len(ds)} slices, {ds.subject_counts()} pacientes, "
            f"label_scheme={ds.label_scheme}"
        )
        print(f"  class_to_idx: {ds.class_to_idx}")
        for cls, n in sorted(ds.class_counts().items(), key=lambda kv: ds.class_to_idx[kv[0]]):
            print(f"    {cls:20s} -> {n:>6d} slices")


def tensor_stats_one_batch(loader: DataLoader, fold: str) -> None:
    print(f"\n=== Estatisticas de 1 batch ({fold}) ===")
    x, y = next(iter(loader))
    print(f"  x.shape = {tuple(x.shape)}, dtype={x.dtype}")
    print(f"  x range = [{x.min().item():.3f}, {x.max().item():.3f}]")
    print(f"  x mean by channel = {x.mean(dim=(0,2,3)).tolist()}")
    print(f"  x std by channel  = {x.std(dim=(0,2,3)).tolist()}")
    print(f"  y.shape = {tuple(y.shape)}, dtype={y.dtype}")
    print(f"  y unique = {sorted(set(y.tolist()))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/eda/manifest.csv"))
    parser.add_argument("--split", type=Path, default=Path("experiments/splits/split_v1.json"))
    parser.add_argument("--label-scheme", choices=["class_3", "class_binary"], default="class_3")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    common = dict(
        manifest_path=args.manifest,
        split_path=args.split,
        label_scheme=args.label_scheme,
        data_root=args.data_root,
        image_size=args.image_size,
    )
    datasets: dict[str, OASISDataset] = {
        "train": OASISDataset(fold="train", **common),
        "val": OASISDataset(fold="val", **common),
        "test": OASISDataset(fold="test", **common),
    }
    print_summary(datasets)

    # Stats em 1 batch do train
    # (este script constrói loaders ad-hoc para inspeção visual; o uso real
    # passa por build_dataloaders em src.data.dataset).
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    tensor_stats_one_batch(train_loader, "train")
    tensor_stats_one_batch(val_loader, "val")

    # Validacao critica: zero overlap de subjects entre folds (redundante com splits.py,
    # mas barato e protege contra futuras edicoes).
    train_subjects = set(datasets["train"]._subjects)
    val_subjects = set(datasets["val"]._subjects)
    test_subjects = set(datasets["test"]._subjects)
    overlap_tv = train_subjects & val_subjects
    overlap_tt = train_subjects & test_subjects
    overlap_vt = val_subjects & test_subjects
    assert not (overlap_tv or overlap_tt or overlap_vt), (
        f"VAZAMENTO entre folds: train∩val={len(overlap_tv)}, "
        f"train∩test={len(overlap_tt)}, val∩test={len(overlap_vt)}"
    )
    print("\nOK: zero overlap de subjects entre folds (verificado em runtime).")

    # Figuras
    grid_per_fold(datasets)
    print(f"Figura por fold salva em {OUT_FIG}")
    augmentation_diff(datasets["train"], datasets["val"])
    print(f"Figura de augmentation salva em {OUT_FIG_AUG}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
