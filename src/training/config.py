"""Configuração de treino serializável (YAML).

Toda decisão fixada em ADR-0008 vira atributo aqui. Cada run salva sua
config completa em `experiments/runs/<run_id>/config.yaml` para auditoria
e reprodutibilidade entre as duas máquinas do aluno.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml

LabelScheme = Literal["class_3", "class_binary"]
ModelName = Literal["resnet50", "vit_base_16", "swin_tiny"]


@dataclass
class TrainConfig:
    """Hiperparâmetros e caminhos de uma run de treino.

    Valores default refletem ADR-0008 (baseline ResNet-50 fine-tune ImageNet).
    """

    # --- Identificação ---
    run_name: str = "resnet50_class3_weighted"
    seed: int = 42

    # --- Dados (ADR-0001, 0002, 0004, 0006) ---
    manifest_path: str = "results/eda/manifest.csv"
    split_path: str = "experiments/splits/split_v1.json"
    data_root: str = "."
    label_scheme: LabelScheme = "class_3"
    image_size: int = 224

    # --- Modelo (ADR-0008) ---
    model_name: ModelName = "resnet50"
    pretrained: bool = True

    # --- Loss (ADR-0007) ---
    use_class_weights: bool = True  # ablação: False = sem peso
    label_smoothing: float = 0.0  # extra ablation

    # --- Optimizer (ADR-0008) ---
    optimizer: Literal["adamw", "sgd"] = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.05
    momentum: float = 0.9  # usado se optimizer=sgd

    # --- Scheduler (ADR-0008) ---
    scheduler: Literal["cosine", "constant"] = "cosine"
    warmup_epochs: int = 1

    # --- Loop (ADR-0008) ---
    epochs: int = 20
    batch_size_train: int = 32
    batch_size_eval: int = 64
    num_workers: int = 0  # Windows/CPU local; aumentar no Colab
    pin_memory: bool = False
    grad_clip_max_norm: float = 1.0
    mixed_precision: bool = False  # off em CPU; True em Colab GPU

    # --- Early stopping (ADR-0008) ---
    early_stopping_metric: Literal["balanced_accuracy", "macro_f1", "val_loss"] = "balanced_accuracy"
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4

    # --- Smoke test (opcional) ---
    smoke_test: bool = False
    smoke_max_batches_per_epoch: int = 50
    smoke_epochs: int = 2

    # --- Saída ---
    output_dir: str = "experiments/runs"

    # --- Metadados preenchidos em runtime ---
    git_commit: str = ""
    started_at: str = ""
    finished_at: str = ""
    device: str = ""
    extra: dict = field(default_factory=dict)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
