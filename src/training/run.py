"""Entrypoint CLI para uma run de treino. Orquestra config → dados → modelo → loop.

Uso (PowerShell, .venv ativo):

    # Smoke test rápido em CPU (2 epochs, 50 batches)
    python -m src.training.run --smoke

    # Treino completo (no Colab) — futuro
    python -m src.training.run --epochs 20 --batch-size 32

    # Ablação sem peso
    python -m src.training.run --no-class-weights --run-name resnet50_class3_noweight

    # Esquema binário
    python -m src.training.run --label-scheme class_binary --run-name resnet50_binary_weighted
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import build_dataloaders
from src.models.factory import MODEL_NAMES, build_model
from src.training.config import TrainConfig
from src.training.loop import fit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2]
        ).decode().strip()
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default=None, help="Nome base da run; default deriva de config.")
    p.add_argument("--model", choices=list(MODEL_NAMES), default="resnet50",
                   help="Arquitetura: resnet50 | vit_base_16 | swin_tiny (ADR-0012).")
    p.add_argument("--drop-path-rate", type=float, default=0.1,
                   help="Stochastic depth (so ViT/Swin; ResNet ignora).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--manifest", default="results/eda/manifest.csv")
    p.add_argument("--split", default="experiments/splits/split_v1.json")
    p.add_argument("--data-root", default=".")
    p.add_argument("--label-scheme", choices=["class_3", "class_binary"], default="class_3")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--no-class-weights", action="store_true", help="Ablacao: sem peso de classe.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batch-size-eval", type=int, default=64)
    # LR: --lr define lr uniforme (V1 compat). --lr-head e --lr-backbone definem LR
    # diferenciado (V2, ADR-0010). Se ambos passados, prevalecem sobre --lr.
    p.add_argument("--lr", type=float, default=1e-4, help="LR uniforme (V1, fallback).")
    p.add_argument("--lr-head", type=float, default=1e-4, help="LR do classifier head (ADR-0010).")
    p.add_argument("--lr-backbone", type=float, default=1e-5, help="LR do backbone (ADR-0010).")
    p.add_argument("--uniform-lr", action="store_true",
                   help="Forca LR uniforme (V1): ignora --lr-head/--lr-backbone, usa --lr.")
    p.add_argument("--weight-decay", type=float, default=0.1, help="ADR-0010 (V1 era 0.05).")
    p.add_argument("--warmup-epochs", type=int, default=2, help="ADR-0010 (V1 era 1).")
    p.add_argument("--drop-rate", type=float, default=0.3, help="Dropout no head (ADR-0010; V1 era 0).")
    p.add_argument("--augment", choices=["light", "strong"], default="strong",
                   help="Augmentation no train (ADR-0010; V1='light', V2='strong').")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=5, help="Early stopping (ADR-0010; V1 era 3).")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 2 epochs, 50 batches.")
    p.add_argument("--smoke-batches", type=int, default=50)
    p.add_argument("--output-dir", default="experiments/runs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    n_classes = 3 if args.label_scheme == "class_3" else 2

    # ADR-0010: LR diferenciado por padrao; --uniform-lr volta ao comportamento V1.
    lr_head_eff = None if args.uniform_lr else args.lr_head
    lr_backbone_eff = None if args.uniform_lr else args.lr_backbone

    cfg = TrainConfig(
        run_name=args.run_name or f"{args.model}_{args.label_scheme}_{'weighted' if not args.no_class_weights else 'noweight'}",
        seed=args.seed,
        manifest_path=args.manifest,
        split_path=args.split,
        data_root=args.data_root,
        label_scheme=args.label_scheme,
        image_size=args.image_size,
        model_name=args.model,
        pretrained=True,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate,
        use_class_weights=not args.no_class_weights,
        optimizer="adamw",
        lr=args.lr,
        lr_head=lr_head_eff,
        lr_backbone=lr_backbone_eff,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        epochs=args.epochs,
        batch_size_train=args.batch_size,
        batch_size_eval=args.batch_size_eval,
        num_workers=args.num_workers,
        augment_strength=args.augment,
        early_stopping_patience=args.patience,
        smoke_test=args.smoke,
        smoke_max_batches_per_epoch=args.smoke_batches,
        output_dir=args.output_dir,
    )

    # ID único do run com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{cfg.run_name}{'_smoke' if cfg.smoke_test else ''}"
    run_dir = Path(cfg.output_dir) / run_id

    cfg.git_commit = get_git_commit()
    cfg.started_at = timestamp
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Run ID: {run_id}")
    print(f"Device: {cfg.device}")
    print(f"Mode: {'SMOKE' if cfg.smoke_test else 'FULL'}")

    # --- Dataloaders ---
    print("Construindo dataloaders...")
    loaders = build_dataloaders(
        manifest_path=cfg.manifest_path,
        split_path=cfg.split_path,
        label_scheme=cfg.label_scheme,
        data_root=cfg.data_root,
        image_size=cfg.image_size,
        batch_size_train=cfg.batch_size_train,
        batch_size_eval=cfg.batch_size_eval,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        train_augment_strength=cfg.augment_strength,
        # Em smoke test, distribuir classes nos primeiros batches para o
        # max_batches da eval não esconder bugs (F-0007 + lição F-0010).
        shuffle_eval=cfg.smoke_test,
        shuffle_eval_seed=cfg.seed,
    )

    class_names = list(loaders["train"].dataset.class_to_idx.keys())  # type: ignore[attr-defined]
    print(f"Classes (idx ordenado): {class_names}")

    # --- Modelo (factory, ADR-0012) ---
    print(f"Construindo modelo {cfg.model_name} (pretrained={cfg.pretrained}, "
          f"drop_rate={cfg.drop_rate}, drop_path_rate={cfg.drop_path_rate})...")
    model, model_info = build_model(
        cfg.model_name,
        num_classes=n_classes,
        pretrained=cfg.pretrained,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
    )
    print(f"  params treinaveis: {model_info['num_params']:,}")

    device = torch.device(cfg.device)
    model = model.to(device)

    # --- Salva config antes de treinar (auditável em caso de crash) ---
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(run_dir / "config.yaml")

    # --- fit ---
    final = fit(
        model=model,
        loaders=loaders,
        cfg=cfg,
        class_names=class_names,
        run_dir=run_dir,
        device=device,
        model_info=model_info,
    )

    cfg.finished_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.to_yaml(run_dir / "config.yaml")

    print(f"\nRun finalizado. Diretorio: {run_dir}")
    print("Metricas finais (test):")
    print(json.dumps(final["test_metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
