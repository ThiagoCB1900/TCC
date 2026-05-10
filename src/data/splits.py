"""Split estratificado por subject ID — OASIS-1 Kaggle.

Implementa a regra crítica nº 1 do CLAUDE.md (split por paciente, nunca por slice)
e a decisão fixada em ADR-0002 (70/15/15, estratificado por class_3, seed fixa).

Saída: JSON em experiments/splits/split_v1.json com listas de subject IDs por
fold, estatísticas por classe e metadados de reprodutibilidade. Pode ser
re-executado em qualquer máquina e produzirá o mesmo split (seed fixa).

Validações automáticas (assert, falha o script se violadas):
- Conjuntos de subjects em train/val/test são disjuntos.
- Cada classe está representada em cada fold (≥ 1 paciente).
- Soma das contagens por fold = total.

Uso (PowerShell, com .venv ativo):
    python -m src.data.splits --manifest results/eda/manifest.csv --out experiments/splits/split_v1.json --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_subject_table(manifest_path: Path) -> pd.DataFrame:
    """Reduz o manifesto (1 linha/slice) a uma tabela com 1 linha por paciente,
    preservando as 3 codificações de classe usadas no projeto."""
    df = pd.read_csv(manifest_path)

    expected = {"subject", "class_4", "class_3", "class_binary"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"manifesto sem colunas esperadas: {missing}")

    # Sanity: cada subject deve ter exatamente uma classe (regra clínica).
    n_classes_per_subj = df.groupby("subject")["class_4"].nunique()
    leaks = n_classes_per_subj[n_classes_per_subj > 1]
    if not leaks.empty:
        raise ValueError(
            f"{len(leaks)} sujeito(s) com múltiplas classes no manifesto — "
            f"manifest corrompido: {leaks.head().to_dict()}"
        )

    by_subject = (
        df.groupby("subject")
        .agg(
            class_4=("class_4", "first"),
            class_3=("class_3", "first"),
            class_binary=("class_binary", "first"),
            n_slices=("path", "count"),
        )
        .reset_index()
        .sort_values("subject")
        .reset_index(drop=True)
    )
    return by_subject


def stratified_subject_split(
    subjects: pd.DataFrame,
    test_size: float,
    val_size: float,
    stratify_col: str,
    seed: int,
) -> dict[str, list[str]]:
    """Divide a tabela de pacientes em três conjuntos disjuntos com estratificação
    pela coluna `stratify_col`.

    Estratégia em 2 passos com `train_test_split`:
      1) separa test do restante;
      2) dentro do restante, separa val.

    Retorna dict com as listas de subject IDs por fold.
    """
    if not (0.0 < test_size < 1.0 and 0.0 < val_size < 1.0):
        raise ValueError("test_size e val_size devem estar em (0, 1).")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size deve ser < 1.")

    # Passo 1: trainval vs test
    trainval, test = train_test_split(
        subjects,
        test_size=test_size,
        random_state=seed,
        stratify=subjects[stratify_col],
    )

    # Passo 2: dentro de trainval, separar val proporcionalmente ao total original
    val_relative = val_size / (1.0 - test_size)
    train, val = train_test_split(
        trainval,
        test_size=val_relative,
        random_state=seed,
        stratify=trainval[stratify_col],
    )

    return {
        "train": sorted(train["subject"].tolist()),
        "val": sorted(val["subject"].tolist()),
        "test": sorted(test["subject"].tolist()),
    }


def validate_splits(splits: dict[str, list[str]], all_subjects: set[str]) -> None:
    """Asserts metodológicos críticos. Falha o programa se violados."""
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])

    assert train.isdisjoint(val), f"train e val sobrepostos: {train & val}"
    assert train.isdisjoint(test), f"train e test sobrepostos: {train & test}"
    assert val.isdisjoint(test), f"val e test sobrepostos: {val & test}"

    union = train | val | test
    missing = all_subjects - union
    extra = union - all_subjects
    assert not missing, f"sujeitos no manifesto fora de qualquer split: {sorted(missing)[:5]}"
    assert not extra, f"sujeitos em algum split fora do manifesto: {sorted(extra)[:5]}"


def per_split_stats(
    by_subject: pd.DataFrame, splits: dict[str, list[str]]
) -> dict[str, dict]:
    """Calcula contagens por classe e por slice em cada fold."""
    stats: dict[str, dict] = {}
    for fold, subj_list in splits.items():
        sub = by_subject[by_subject["subject"].isin(subj_list)]
        stats[fold] = {
            "n_subjects": int(len(sub)),
            "n_slices": int(sub["n_slices"].sum()),
            "by_class_4": sub["class_4"].value_counts().to_dict(),
            "by_class_3": sub["class_3"].value_counts().to_dict(),
            "by_class_binary": sub["class_binary"].value_counts().to_dict(),
            "slices_by_class_3": sub.groupby("class_3")["n_slices"].sum().to_dict(),
        }
    return stats


def assert_class_coverage(stats: dict[str, dict], stratify_col: str) -> None:
    """Cada classe deve aparecer em todos os folds (≥1 paciente)."""
    by_class_key = f"by_{stratify_col}"
    classes_seen: set[str] = set()
    for fold_stats in stats.values():
        classes_seen.update(fold_stats[by_class_key].keys())

    for fold, fold_stats in stats.items():
        for cls in classes_seen:
            n = fold_stats[by_class_key].get(cls, 0)
            assert n > 0, (
                f"fold '{fold}' não tem nenhum paciente da classe '{cls}' "
                f"(estratificação por '{stratify_col}' falhou)."
            )


def write_split_json(
    out_path: Path,
    splits: dict[str, list[str]],
    stats: dict[str, dict],
    by_subject: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    payload = {
        "version": out_path.stem,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "ratios": {
            "train": round(1.0 - args.test_size - args.val_size, 4),
            "val": args.val_size,
            "test": args.test_size,
        },
        "stratified_by": args.stratify_col,
        "manifest": str(args.manifest).replace("\\", "/"),
        "n_subjects_total": int(len(by_subject)),
        "n_slices_total": int(by_subject["n_slices"].sum()),
        "splits": {
            fold: {
                **stats[fold],
                "subjects": splits[fold],
            }
            for fold in ("train", "val", "test")
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(stats: dict[str, dict], stratify_col: str) -> None:
    print("\n=== Resumo do split ===")
    rows = []
    by_class_key = f"by_{stratify_col}"
    classes = sorted(
        {c for fold_stats in stats.values() for c in fold_stats[by_class_key]}
    )
    for fold in ("train", "val", "test"):
        s = stats[fold]
        row = {
            "fold": fold,
            "n_subjects": s["n_subjects"],
            "n_slices": s["n_slices"],
        }
        for cls in classes:
            row[cls] = s[by_class_key].get(cls, 0)
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("fold")
    print(summary.to_string())

    # proporções relativas por classe — checar se ficaram próximas
    print(f"\nProporção de sujeitos por classe ({stratify_col}):")
    proportion_rows = []
    for fold in ("train", "val", "test"):
        s = stats[fold]
        total = s["n_subjects"]
        proportion_rows.append(
            {
                "fold": fold,
                **{cls: f"{s[by_class_key].get(cls, 0)/total:.1%}" for cls in classes},
            }
        )
    print(pd.DataFrame(proportion_rows).set_index("fold").to_string())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("results/eda/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("experiments/splits/split_v1.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument(
        "--stratify-col",
        choices=["class_3", "class_4", "class_binary"],
        default="class_3",
        help="coluna usada para estratificação (default: class_3, ver ADR-0001).",
    )
    args = parser.parse_args(argv)

    if not args.manifest.exists():
        print(f"[ERRO] manifesto não encontrado: {args.manifest.resolve()}", file=sys.stderr)
        return 2

    print(f"Lendo manifesto: {args.manifest}")
    by_subject = load_subject_table(args.manifest)
    print(f"  {len(by_subject)} sujeitos únicos, {int(by_subject['n_slices'].sum())} slices.")

    print(f"Estratificando por '{args.stratify_col}' com seed={args.seed} ...")
    splits = stratified_subject_split(
        by_subject,
        test_size=args.test_size,
        val_size=args.val_size,
        stratify_col=args.stratify_col,
        seed=args.seed,
    )

    validate_splits(splits, set(by_subject["subject"]))

    stats = per_split_stats(by_subject, splits)
    assert_class_coverage(stats, args.stratify_col)

    write_split_json(args.out, splits, stats, by_subject, args)
    print_summary(stats, args.stratify_col)

    print(f"\nOK. Split salvo em {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
