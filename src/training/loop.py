"""Loop de treino — PyTorch puro, conforme ADR-0008.

Funções principais:
  - `compute_class_weights(loader)` — ADR-0007, pesos 'balanced' sobre o split de treino.
  - `train_one_epoch(...)` — uma época de treino com weighted CE + clipping + scheduler.
  - `evaluate(...)` — eval determinística sobre um loader (model.eval + no_grad).
  - `fit(...)` — orquestra todas as épocas, early stopping, salvamento de checkpoints,
    history JSON, métricas finais no test.

Nada de "magia": cada passo é explícito e auditável.
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import (
    ClassificationResult,
    collect_predictions,
    compute_metrics,
    format_metrics_table,
)
from src.training.config import TrainConfig


# ---------------------------------------------------------------------------
# Pesos de classe (ADR-0007)
# ---------------------------------------------------------------------------

def make_param_groups(
    model: torch.nn.Module,
    lr_head: float,
    lr_backbone: float,
    weight_decay: float,
) -> list[dict]:
    """Separa parâmetros do classifier head dos do backbone para LR diferenciado.

    ADR-0010: backbone pretrained ImageNet só precisa ajuste fino; head reinicializado
    precisa aprender do zero. Usa `model.get_classifier()` que timm padroniza para
    qualquer arquitetura (ResNet, ViT, Swin) — identifica o head por identidade do
    tensor de parâmetro (`id`), robusto a renomeações.
    """
    classifier_ids = {id(p) for p in model.get_classifier().parameters()}
    head: list[torch.nn.Parameter] = []
    backbone: list[torch.nn.Parameter] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (head if id(p) in classifier_ids else backbone).append(p)
    if not head:
        raise RuntimeError(
            "make_param_groups: nenhum parâmetro identificado como classifier head. "
            "Verifique `model.get_classifier()` da arquitetura."
        )
    return [
        {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay},
    ]


def compute_class_weights(train_loader: DataLoader, num_classes: int) -> torch.Tensor:
    """Calcula pesos 'balanced' (sklearn formula) sobre o split de treino.

    weight_c = n_total / (n_classes * n_train_in_class_c)

    Conta os labels uma vez (varre o dataset por baixo, não o loader, para evitar
    augmentation aleatória). Trata classe ausente como peso 1.0 (não deveria ocorrer
    se assert_class_coverage passou em splits.py).
    """
    # Acessa diretamente o dataset subjacente — evita rodar augmentations e
    # garante determinismo. Atributos `_labels` são públicos para o caller
    # (ver convenção no OASISDataset).
    ds = train_loader.dataset
    labels: np.ndarray = ds._labels  # type: ignore[attr-defined]
    counts = Counter(labels.tolist())
    total = len(labels)
    weights = []
    for c in range(num_classes):
        n_c = counts.get(c, 0)
        if n_c == 0:
            weights.append(1.0)  # fallback defensivo
        else:
            weights.append(total / (num_classes * n_c))
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# LR scheduler (warmup + cosine), conforme ADR-0008
# ---------------------------------------------------------------------------

def make_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup linear de `warmup_epochs` + cosine annealing até o fim."""
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if cfg.scheduler == "constant":
            return 1.0
        # cosine
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Uma época de treino
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_fn: torch.nn.Module,
    device: torch.device,
    grad_clip_max_norm: float,
    epoch: int,
    total_epochs: int,
    max_batches: int | None = None,
) -> dict:
    """Treina uma época. Retorna dict com loss média e nº de batches efetivos.

    `max_batches` limita iterações (usado no smoke test).
    """
    model.train()
    losses: list[float] = []

    iter_loader = enumerate(loader)
    desc = f"epoch {epoch}/{total_epochs} train"
    pbar = tqdm(iter_loader, total=max_batches or len(loader), desc=desc, leave=False)

    for i, (x, y) in pbar:
        if max_batches is not None and i >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        if grad_clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "n_batches": len(losses),
    }


# ---------------------------------------------------------------------------
# Eval sobre um loader inteiro
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    class_names: list[str],
    max_batches: int | None = None,
) -> tuple[ClassificationResult, float]:
    """Eval determinística: model.eval() + no_grad, métricas completas + loss média."""
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []
    losses: list[float] = []

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            losses.append(loss_fn(logits, y).item())
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
            pred = proba.argmax(axis=-1)
            all_true.append(y.cpu().numpy())
            all_pred.append(pred)
            all_proba.append(proba)

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_proba = np.concatenate(all_proba)
    result = compute_metrics(y_true, y_pred, y_proba, class_names)
    return result, float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# fit: orquestrador completo (treino + val + early stopping + checkpoint)
# ---------------------------------------------------------------------------

def fit(
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    cfg: TrainConfig,
    class_names: list[str],
    run_dir: Path,
    device: torch.device,
    model_info: dict,
) -> dict:
    """Loop completo. Salva config, checkpoint best, history.json, final_test_metrics.json.

    Retorna dict com métricas finais no test set e caminho do best checkpoint.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"== Run dir: {run_dir} ==")
    log(f"Model: {model_info}")
    log(f"Device: {device}")
    log(f"Config:\n{json.dumps(asdict(cfg), indent=2, ensure_ascii=False)}")

    # --- Pesos de classe (ADR-0007) ---
    if cfg.use_class_weights:
        weights = compute_class_weights(loaders["train"], num_classes=len(class_names))
        log(f"Pesos de classe (ADR-0007): {weights.tolist()}")
    else:
        weights = None
        log("Sem peso de classe (ablacao sem peso).")

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None,
        label_smoothing=cfg.label_smoothing,
    )

    # --- Optimizer ---
    # ADR-0010: se lr_head e lr_backbone foram especificados, usa LR diferenciado
    # via param_groups. Caso contrario, usa lr uniforme (compatibilidade V1).
    use_differentiated_lr = cfg.lr_head is not None and cfg.lr_backbone is not None

    if cfg.optimizer == "adamw":
        if use_differentiated_lr:
            param_groups = make_param_groups(
                model, cfg.lr_head, cfg.lr_backbone, cfg.weight_decay
            )
            optimizer = torch.optim.AdamW(param_groups)
            log(
                f"LR diferenciado (ADR-0010): backbone={cfg.lr_backbone}, "
                f"head={cfg.lr_head}, weight_decay={cfg.weight_decay}"
            )
            log(
                f"  parametros backbone: {sum(p.numel() for p in param_groups[0]['params']):,}, "
                f"head: {sum(p.numel() for p in param_groups[1]['params']):,}"
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            log(f"LR uniforme (V1): lr={cfg.lr}, weight_decay={cfg.weight_decay}")
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"optimizer nao suportado: {cfg.optimizer}")

    # --- Scheduler ---
    steps_per_epoch = (
        cfg.smoke_max_batches_per_epoch if cfg.smoke_test else len(loaders["train"])
    )
    scheduler = make_scheduler(optimizer, cfg, steps_per_epoch)

    # --- Loop de épocas ---
    epochs = cfg.smoke_epochs if cfg.smoke_test else cfg.epochs
    max_batches = cfg.smoke_max_batches_per_epoch if cfg.smoke_test else None

    history: list[dict] = []
    best_metric = -math.inf
    best_epoch = -1
    patience_left = cfg.early_stopping_patience
    best_ckpt = run_dir / "checkpoint_best.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_out = train_one_epoch(
            model, loaders["train"], optimizer, scheduler, loss_fn, device,
            cfg.grad_clip_max_norm, epoch, epochs, max_batches=max_batches,
        )

        val_result, val_loss = evaluate(
            model, loaders["val"], loss_fn, device, class_names,
            max_batches=max_batches,
        )

        epoch_time = time.time() - epoch_start
        log(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss={train_out['loss']:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_bal_acc={val_result.balanced_accuracy:.4f}  "
            f"val_macro_f1={val_result.macro_f1:.4f}  "
            f"time={epoch_time:.1f}s"
        )

        # Métrica monitorada
        if cfg.early_stopping_metric == "balanced_accuracy":
            metric_val = val_result.balanced_accuracy
        elif cfg.early_stopping_metric == "macro_f1":
            metric_val = val_result.macro_f1
        elif cfg.early_stopping_metric == "val_loss":
            metric_val = -val_loss  # minimizar -> maximizar negativo
        else:
            raise ValueError(cfg.early_stopping_metric)

        improved = metric_val > best_metric + cfg.early_stopping_min_delta

        history.append({
            "epoch": epoch,
            "train_loss": train_out["loss"],
            "train_n_batches": train_out["n_batches"],
            "val_loss": val_loss,
            "val_metrics": val_result.to_dict(),
            "lr": scheduler.get_last_lr()[0],
            "improved": improved,
            "epoch_time_seconds": epoch_time,
        })

        # Salva history a cada época (auditável mesmo em caso de crash)
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if improved:
            best_metric = metric_val
            best_epoch = epoch
            patience_left = cfg.early_stopping_patience
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": asdict(cfg),
                    "model_info": model_info,
                    "best_metric": best_metric,
                    "metric_name": cfg.early_stopping_metric,
                },
                best_ckpt,
            )
            log(f"  -> checkpoint_best salvo (epoch={epoch}, metric={best_metric:.4f})")
        else:
            patience_left -= 1
            log(f"  patience restante: {patience_left}")
            if patience_left <= 0:
                log("Early stopping disparou.")
                break

    total_time = time.time() - t0
    log(f"Treino concluido em {total_time:.1f}s. Best epoch={best_epoch} ({cfg.early_stopping_metric}={best_metric:.4f}).")

    # --- Avaliação final no test set (carrega best) ---
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        log("Best checkpoint recarregado para avaliacao final.")

    test_result, test_loss = evaluate(
        model, loaders["test"], loss_fn, device, class_names,
        max_batches=max_batches,
    )
    log("Metricas finais no TEST:")
    log(format_metrics_table(test_result))

    final = {
        "best_epoch": best_epoch,
        "best_metric_name": cfg.early_stopping_metric,
        "best_metric_val_value": best_metric,
        "test_loss": test_loss,
        "test_metrics": test_result.to_dict(),
        "total_training_time_seconds": total_time,
        "checkpoint_best": str(best_ckpt),
    }
    (run_dir / "final_test_metrics.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return final


__all__ = ["compute_class_weights", "train_one_epoch", "evaluate", "fit"]
