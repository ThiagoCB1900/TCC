"""Métricas de classificação para o TCC.

Métricas primárias (ADR-0005):
- macro-F1
- balanced accuracy
- AUC macro (one-vs-rest)

Adicionais:
- accuracy bruta (apenas para comparabilidade com RanCom-ViT — secundária)
- F1, precision, recall por classe
- matriz de confusão
- teste de McNemar (par-a-par entre modelos)

Implementação usa torchmetrics (já no requirements) para o cálculo principal e
sklearn para matriz de confusão / McNemar. Todas as métricas são computadas
sobre **as previsões inteiras do fold** (não médias de batch) — ver F-0007
para por que isso importa em val/test sem shuffle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torchmetrics.functional import accuracy as tm_accuracy


@dataclass
class ClassificationResult:
    """Métricas computadas para um fold (val ou test) numa única época/run.

    Atributos:
        accuracy: acurácia bruta (secundária)
        balanced_accuracy: média da recall por classe (primária)
        macro_f1: F1 não-ponderada entre classes (primária)
        auc_macro: AUC macro one-vs-rest (primária; requer probabilidades)
        f1_per_class: F1 por classe (lista ordenada por idx_to_class)
        precision_per_class: idem
        recall_per_class: idem
        confusion: matriz de confusão NxN (linhas = real, colunas = predito)
        n_samples: número de previsões usadas para o cálculo
        class_names: nomes das classes na ordem dos índices
    """

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    auc_macro: float | None
    f1_per_class: list[float]
    precision_per_class: list[float]
    recall_per_class: list[float]
    confusion: list[list[int]]
    n_samples: int
    class_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "auc_macro": self.auc_macro,
            "f1_per_class": dict(zip(self.class_names, self.f1_per_class)),
            "precision_per_class": dict(zip(self.class_names, self.precision_per_class)),
            "recall_per_class": dict(zip(self.class_names, self.recall_per_class)),
            "confusion": self.confusion,
            "class_names": self.class_names,
            "n_samples": self.n_samples,
        }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    class_names: Sequence[str],
) -> ClassificationResult:
    """Computa todas as métricas da ADR-0005 sobre vetores de previsão completos.

    Args:
        y_true: shape (N,), labels inteiros 0..C-1.
        y_pred: shape (N,), labels preditos.
        y_proba: shape (N, C) com probabilidades softmax; None pula AUC.
        class_names: lista de nomes na ordem dos índices.

    Returns:
        ClassificationResult com todas as métricas preenchidas (AUC None se y_proba=None).
    """
    n_classes = len(class_names)

    accuracy = float((y_true == y_pred).mean())
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0
    ).tolist()
    precision_per_class = precision_score(
        y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0
    ).tolist()
    recall_per_class = recall_score(
        y_true, y_pred, average=None, labels=list(range(n_classes)), zero_division=0
    ).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()

    auc_macro: float | None = None
    if y_proba is not None and n_classes >= 2:
        # AUC requer pelo menos 1 amostra de cada classe presente em y_true
        present = set(np.unique(y_true).tolist())
        if len(present) >= 2 and present == set(range(n_classes)):
            try:
                auc_macro = float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="macro",
                        labels=list(range(n_classes)),
                    )
                )
            except ValueError:
                auc_macro = None

    return ClassificationResult(
        accuracy=accuracy,
        balanced_accuracy=bal_acc,
        macro_f1=macro_f1,
        auc_macro=auc_macro,
        f1_per_class=[float(x) for x in f1_per_class],
        precision_per_class=[float(x) for x in precision_per_class],
        recall_per_class=[float(x) for x in recall_per_class],
        confusion=cm,
        n_samples=int(len(y_true)),
        class_names=list(class_names),
    )


def format_metrics_table(result: ClassificationResult) -> str:
    """Tabela legível para impressão no terminal."""
    lines = [
        f"  accuracy           : {result.accuracy:.4f}",
        f"  balanced_accuracy  : {result.balanced_accuracy:.4f}  (primaria)",
        f"  macro_f1           : {result.macro_f1:.4f}  (primaria)",
    ]
    if result.auc_macro is not None:
        lines.append(f"  auc_macro          : {result.auc_macro:.4f}  (primaria)")
    else:
        lines.append("  auc_macro          : N/A (faltam classes em y_true)")
    lines.append(f"  n_samples          : {result.n_samples}")
    lines.append("  por classe:")
    for i, cls in enumerate(result.class_names):
        lines.append(
            f"    [{i}] {cls:20s}  P={result.precision_per_class[i]:.3f}  "
            f"R={result.recall_per_class[i]:.3f}  F1={result.f1_per_class[i]:.3f}"
        )
    lines.append("  matriz de confusao (real x predito):")
    header = "        " + " ".join(f"{cls[:10]:>10s}" for cls in result.class_names)
    lines.append(header)
    for i, row in enumerate(result.confusion):
        cells = " ".join(f"{v:>10d}" for v in row)
        lines.append(f"    {result.class_names[i][:6]:>6s}  {cells}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# McNemar — comparação par-a-par entre dois modelos no mesmo test set.
# ADR-0005 requer este teste para validar que diferenças entre modelos
# são estatisticamente significativas.
# ----------------------------------------------------------------------

def mcnemar_test(
    y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray
) -> dict[str, float]:
    """Teste de McNemar entre dois classificadores no mesmo test set.

    H0: as taxas de erro dos dois modelos são iguais.
    Usa correção de continuidade (sem versão exata, suficiente para N grande).

    Returns:
        dict com b (acerto B & erro A), c (acerto A & erro B), statistic, p_value
    """
    from scipy.stats import chi2

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    b = int(np.sum(~correct_a & correct_b))  # A errou, B acertou
    c = int(np.sum(correct_a & ~correct_b))  # A acertou, B errou

    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}

    # Estatística com correção de continuidade
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1.0 - chi2.cdf(statistic, df=1))
    return {"b": b, "c": c, "statistic": float(statistic), "p_value": p_value}


# ----------------------------------------------------------------------
# Helper: roda 1 modelo sobre 1 dataloader e devolve y_true / y_pred / y_proba.
# Usado por evaluate() do loop e por scripts de auditoria.
# ----------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roda model.eval() sobre o loader inteiro e retorna (y_true, y_pred, y_proba)."""
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
        pred = proba.argmax(axis=-1)
        all_true.append(y.numpy())
        all_pred.append(pred)
        all_proba.append(proba)
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_proba),
    )
