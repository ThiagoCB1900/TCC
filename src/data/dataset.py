"""Dataset PyTorch para slices axiais OASIS-1 (versão Kaggle pré-processada).

Encapsula toda a metodologia de carga e pré-processamento decidida em ADRs:

- ADR-0001 — esquema de 3 classes (Mild+Moderate fundidos), com fallback binário.
- ADR-0002 — split por paciente lido de experiments/splits/split_v1.json.
- ADR-0004 — resize squash 224×224, RGB sintético mantido, normalização ImageNet.
- ADR-0006 — augmentations leves só no train; label encoding por severidade clínica.

Uso típico:
    from src.data.dataset import OASISDataset, build_dataloaders

    train_ds = OASISDataset(
        manifest_path="results/eda/manifest.csv",
        split_path="experiments/splits/split_v1.json",
        fold="train",
        label_scheme="class_3",
        data_root=".",
    )
    print(train_ds.class_to_idx)  # {'non_demented': 0, 'very_mild': 1, 'mild_or_moderate': 2}
    loaders = build_dataloaders(...)  # train + val + test
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

# --- Constantes de domínio (ADR-0006) ---------------------------------------

# Ordem por severidade clínica, hardcoded.
CLASS_TO_IDX_3 = {
    "non_demented": 0,
    "very_mild": 1,
    "mild_or_moderate": 2,
}
CLASS_TO_IDX_BINARY = {
    "non_demented": 0,
    "demented": 1,
}

LabelScheme = Literal["class_3", "class_binary"]
Fold = Literal["train", "val", "test"]

# Estatísticas ImageNet — ADR-0004 (compatibilidade com pesos pré-treinados timm).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Tamanho de entrada — ADR-0004.
DEFAULT_IMAGE_SIZE = 224


# --- Transforms -------------------------------------------------------------

@dataclass(frozen=True)
class TransformConfig:
    """Configuração de transforms por fold."""

    image_size: int = DEFAULT_IMAGE_SIZE
    augment: bool = False  # True só para o fold de treino


def build_transform(cfg: TransformConfig) -> T.Compose:
    """Pipeline determinístico (val/test) ou com augmentation leve (train).

    Ordem importa:
      1. Resize squash para garantir tamanho fixo (deformação horizontal aceita;
         ver ADR-0004).
      2. Augmentations geométricas/colorimétricas, se aplicável (train apenas;
         ver ADR-0006).
      3. ToImage + ToDtype para tensor float32 em [0,1].
      4. Normalize ImageNet — última etapa, sempre.
    """
    steps: list = [T.Resize((cfg.image_size, cfg.image_size), antialias=True)]

    if cfg.augment:
        steps.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=5, fill=0),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )

    steps.extend(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return T.Compose(steps)


# --- Dataset ----------------------------------------------------------------

class OASISDataset(Dataset):
    """Slices 2D OASIS-1 (Kaggle) filtrados pelo split por paciente.

    Não realiza split internamente — recebe o `split_path` (JSON gerado por
    `src/data/splits.py`) e usa apenas os subjects do `fold` solicitado. Isso
    garante que o split metodológico fique versionado e auditável (ADR-0002).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split_path: str | Path,
        fold: Fold,
        label_scheme: LabelScheme = "class_3",
        data_root: str | Path = ".",
        image_size: int = DEFAULT_IMAGE_SIZE,
        augment: bool | None = None,
    ) -> None:
        super().__init__()

        if fold not in ("train", "val", "test"):
            raise ValueError(f"fold deve ser 'train', 'val' ou 'test'; recebido: {fold!r}")
        if label_scheme not in ("class_3", "class_binary"):
            raise ValueError(
                f"label_scheme deve ser 'class_3' ou 'class_binary'; recebido: {label_scheme!r}"
            )

        manifest_path = Path(manifest_path)
        split_path = Path(split_path)
        data_root = Path(data_root)

        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifesto não encontrado: {manifest_path.resolve()}")
        if not split_path.is_file():
            raise FileNotFoundError(f"split JSON não encontrado: {split_path.resolve()}")

        # Carrega manifesto e filtra pelo fold via lista de subjects do split.
        manifest = pd.read_csv(manifest_path)
        with split_path.open(encoding="utf-8") as f:
            split_payload = json.load(f)
        subjects_in_fold = set(split_payload["splits"][fold]["subjects"])

        df = manifest[manifest["subject"].isin(subjects_in_fold)].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"nenhuma linha do manifesto corresponde ao fold {fold!r}; "
                f"verifique se manifest e split foram gerados a partir do mesmo Data/."
            )

        # Validação de integridade: todos os subjects do split devem ter slices no manifesto.
        missing = subjects_in_fold - set(df["subject"])
        if missing:
            raise ValueError(
                f"split {fold!r} referencia {len(missing)} subjects ausentes do manifesto: "
                f"{sorted(missing)[:3]}..."
            )

        # Mapeamento label → int (ADR-0006).
        self.class_to_idx: dict[str, int] = (
            CLASS_TO_IDX_3 if label_scheme == "class_3" else CLASS_TO_IDX_BINARY
        )
        self.idx_to_class: dict[int, str] = {v: k for k, v in self.class_to_idx.items()}

        # Pré-calcula labels inteiros para evitar lookup por amostra.
        labels = df[label_scheme].map(self.class_to_idx)
        if labels.isna().any():
            invalid = df.loc[labels.isna(), label_scheme].unique().tolist()
            raise ValueError(
                f"manifesto contém valores em '{label_scheme}' fora do mapeamento: {invalid}"
            )
        self._labels = labels.astype("int64").to_numpy()
        self._paths = df["path"].to_numpy()
        self._subjects = df["subject"].to_numpy()
        self.data_root = data_root
        self.fold = fold
        self.label_scheme = label_scheme

        # Augmentation: por default, ligado se fold=='train'; pode ser sobrescrito.
        if augment is None:
            augment = fold == "train"
        self.transform = build_transform(TransformConfig(image_size=image_size, augment=augment))

    # ------------------------- API PyTorch -------------------------

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rel_path = self._paths[idx]
        full_path = self.data_root / rel_path
        with Image.open(full_path) as img:
            img = img.convert("RGB")  # garante 3 canais (já são, mas explícito)
            tensor = self.transform(img)
        label = int(self._labels[idx])
        return tensor, label

    # ------------------------- Inspeção -------------------------

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def class_counts(self) -> dict[str, int]:
        """Quantos slices por classe textual neste fold."""
        counts = pd.Series(self._labels).value_counts().to_dict()
        return {self.idx_to_class[k]: int(v) for k, v in counts.items()}

    def subject_counts(self) -> int:
        """Quantos pacientes únicos neste fold."""
        return int(pd.Series(self._subjects).nunique())


# --- DataLoader builder -----------------------------------------------------

def build_dataloaders(
    manifest_path: str | Path,
    split_path: str | Path,
    *,
    label_scheme: LabelScheme = "class_3",
    data_root: str | Path = ".",
    image_size: int = DEFAULT_IMAGE_SIZE,
    batch_size_train: int = 32,
    batch_size_eval: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_eval: bool = False,
    shuffle_eval_seed: int = 42,
) -> dict[Fold, DataLoader]:
    """Constrói os 3 DataLoaders (train/val/test).

    Notas:
      - `shuffle=True` sempre no train; val/test seguem `shuffle_eval`.
      - `shuffle_eval=False` (default, padrão de avaliação determinística) faz
        os batches de val/test virem na ordem do manifesto. **Atenção F-0007:**
        o manifesto está agrupado por classe; quando combinado com avaliação
        truncada (max_batches) isso pode esconder bugs. **Sempre que truncar
        eval (ex: smoke test), passar `shuffle_eval=True`** para distribuir
        as classes nos primeiros batches.
      - `shuffle_eval` usa um `torch.Generator` com seed fixa: a ordem é
        embaralhada **uma vez**, e cada época percorre essa mesma ordem
        embaralhada — preserva determinismo na avaliação completa.
      - Augmentation só no train (default do OASISDataset).
      - num_workers=0 por default para Windows/CPU local; aumentar no Colab.
      - pin_memory=False local; True no Colab com GPU.
    """
    common = dict(
        manifest_path=manifest_path,
        split_path=split_path,
        label_scheme=label_scheme,
        data_root=data_root,
        image_size=image_size,
    )

    datasets: dict[Fold, OASISDataset] = {
        "train": OASISDataset(fold="train", **common),
        "val": OASISDataset(fold="val", **common),
        "test": OASISDataset(fold="test", **common),
    }

    eval_gen = torch.Generator()
    eval_gen.manual_seed(shuffle_eval_seed)

    loaders: dict[Fold, DataLoader] = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size_train,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size_eval,
            shuffle=shuffle_eval,
            generator=eval_gen if shuffle_eval else None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size_eval,
            shuffle=shuffle_eval,
            generator=eval_gen if shuffle_eval else None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    }
    return loaders
