"""Re-avalia um checkpoint no test set: salva test_predictions.npz + métricas.

Útil para runs antigas que não persistiram predições (ex: ResNet V2, treinado
antes da persistência no loop) — permite McNemar entre todas as arquiteturas
sem re-treinar. Roda em CPU local.

Uso:
    python -m src.evaluation.eval_checkpoint \
        --run-dir experiments/runs/20260522_013246_resnet50_v2_class3_weighted_kaggle
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import OASISDataset
from src.evaluation.metrics import collect_predictions, compute_metrics, format_metrics_table
from src.models.factory import build_model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=Path("results/eda/manifest.csv"))
    p.add_argument("--split", type=Path, default=Path("experiments/splits/split_v1.json"))
    p.add_argument("--data-root", type=Path, default=Path("."))
    p.add_argument("--label-scheme", default="class_3")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)

    ckpt_path = args.run_dir / "checkpoint_best.pt"
    if not ckpt_path.is_file():
        print(f"[ERRO] checkpoint nao encontrado: {ckpt_path}")
        return 2

    device = torch.device("cpu")
    n_classes = 3 if args.label_scheme == "class_3" else 2
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    info = ckpt.get("model_info", {})
    name = info.get("name") or cfg.get("model_name")
    model, _ = build_model(name, num_classes=n_classes, pretrained=False,
                           drop_rate=cfg.get("drop_rate", 0.0),
                           drop_path_rate=cfg.get("drop_path_rate", 0.0))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Modelo: {name}")

    ds = OASISDataset(args.manifest, args.split, fold="test",
                      label_scheme=args.label_scheme, data_root=args.data_root,
                      augment_strength="off")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    class_names = list(ds.class_to_idx.keys())

    print(f"Avaliando {len(ds)} slices do test...")
    y_true, y_pred, y_proba = collect_predictions(model, loader, device)
    result = compute_metrics(y_true, y_pred, y_proba, class_names)
    print(format_metrics_table(result))

    np.savez_compressed(
        args.run_dir / "test_predictions.npz",
        y_true=y_true, y_pred=y_pred, y_proba=y_proba,
        subjects=ds._subjects, class_names=np.array(class_names),
    )
    (args.run_dir / "reeval_test_metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Predicoes salvas em {args.run_dir / 'test_predictions.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
