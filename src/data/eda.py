"""
EDA — OASIS-1 (versão Kaggle pré-processada, slices axiais 2D em JPG).

Gera manifesto reproduzível + estatísticas + figuras em results/eda/.

Uso (PowerShell, com .venv ativo):
    python -m src.data.eda --data-root Data --out-dir results/eda

Saídas:
    manifest.csv        — uma linha por slice, com classe e metadados parseados
    summary.json        — estatísticas agregadas por classe
    eda_report.md       — relatório legível
    figures/*.png       — gráficos
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configurações de domínio — alinhadas ao CLAUDE.md
# ---------------------------------------------------------------------------

CLASS_FOLDERS = {
    "Non Demented": "non_demented",
    "Very mild Dementia": "very_mild",
    "Mild Dementia": "mild",
    "Moderate Dementia": "moderate",
}

# Esquema de 3 classes definido com o aluno (Moderate fundido com Mild
# por ter apenas 2 pacientes — split por paciente seria inviável isolado).
CLASS_3 = {
    "non_demented": "non_demented",
    "very_mild": "very_mild",
    "mild": "mild_or_moderate",
    "moderate": "mild_or_moderate",
}

# Esquema binário (Demented vs Non Demented) — fallback alinhado ao plano original.
CLASS_BINARY = {
    "non_demented": "non_demented",
    "very_mild": "demented",
    "mild": "demented",
    "moderate": "demented",
}

FILENAME_RE = re.compile(
    r"^OAS1_(?P<subject>\d{4})_MR(?P<session>\d+)_mpr-(?P<mpr>\d+)_(?P<slice_idx>\d+)\.jpg$"
)


@dataclass(frozen=True)
class Paths:
    data_root: Path
    out_dir: Path
    figures: Path

    @classmethod
    def build(cls, data_root: Path, out_dir: Path) -> "Paths":
        figures = out_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        return cls(data_root=data_root, out_dir=out_dir, figures=figures)


# ---------------------------------------------------------------------------
# 1. Construção do manifesto (uma linha por slice)
# ---------------------------------------------------------------------------

def parse_filename(name: str) -> dict | None:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    g = m.groupdict()
    return {
        "subject": f"OAS1_{g['subject']}",
        "session": int(g["session"]),
        "mpr": int(g["mpr"]),
        "slice_idx": int(g["slice_idx"]),
    }


def build_manifest(data_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    bad: list[str] = []
    for folder_name, label_4 in CLASS_FOLDERS.items():
        folder = data_root / folder_name
        if not folder.exists():
            print(f"[AVISO] pasta ausente: {folder}", file=sys.stderr)
            continue
        for jpg in folder.iterdir():
            if jpg.suffix.lower() != ".jpg":
                continue
            parsed = parse_filename(jpg.name)
            if parsed is None:
                bad.append(str(jpg))
                continue
            rows.append(
                {
                    "path": str(jpg.relative_to(data_root.parent)).replace("\\", "/"),
                    "class_4": label_4,
                    "class_3": CLASS_3[label_4],
                    "class_binary": CLASS_BINARY[label_4],
                    **parsed,
                }
            )
    if bad:
        print(f"[AVISO] {len(bad)} arquivos com nome fora do padrão (ignorados).", file=sys.stderr)
        for b in bad[:5]:
            print(f"  - {b}", file=sys.stderr)
    df = pd.DataFrame(rows)
    df.sort_values(["class_4", "subject", "mpr", "slice_idx"], inplace=True, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# 2. Sanity checks
# ---------------------------------------------------------------------------

def sanity_checks(df: pd.DataFrame) -> dict:
    report: dict = {}

    # 2.1 — um sujeito não pode aparecer em mais de uma classe (regra clínica)
    by_subject = df.groupby("subject")["class_4"].nunique()
    leaks = by_subject[by_subject > 1]
    report["subjects_with_multiple_labels"] = leaks.to_dict()
    if not leaks.empty:
        print(f"[ALERTA] {len(leaks)} sujeitos rotulados em >1 classe:", file=sys.stderr)
        print(leaks.to_string(), file=sys.stderr)

    # 2.2 — sessões esperadas
    report["unique_sessions"] = sorted(df["session"].unique().tolist())

    # 2.3 — mpr range
    report["mpr_range"] = [int(df["mpr"].min()), int(df["mpr"].max())]
    report["mpr_unique"] = sorted(df["mpr"].unique().tolist())

    # 2.4 — slice index range global e por classe
    report["slice_idx_range_global"] = [int(df["slice_idx"].min()), int(df["slice_idx"].max())]
    report["slice_idx_range_by_class"] = {
        c: [int(g["slice_idx"].min()), int(g["slice_idx"].max())]
        for c, g in df.groupby("class_4")
    }

    return report


# ---------------------------------------------------------------------------
# 3. Estatísticas agregadas
# ---------------------------------------------------------------------------

def class_summary(df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    g = df.groupby(class_col)
    out = pd.DataFrame(
        {
            "n_subjects": g["subject"].nunique(),
            "n_acquisitions": g.apply(
                lambda x: x[["subject", "session", "mpr"]].drop_duplicates().shape[0],
                include_groups=False,
            ),
            "n_slices": g.size(),
        }
    )
    out["pct_slices"] = (out["n_slices"] / out["n_slices"].sum() * 100).round(2)
    out["slices_per_subject_mean"] = (out["n_slices"] / out["n_subjects"]).round(1)
    return out.sort_values("n_slices", ascending=False)


# ---------------------------------------------------------------------------
# 4. Amostragem de imagens — checagem de resolução, modo, intensidade
# ---------------------------------------------------------------------------

def sample_image_stats(df: pd.DataFrame, data_root: Path, n_per_class: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    rows: list[dict] = []
    for cls, group in df.groupby("class_4"):
        sample_idx = rng.choice(len(group), size=min(n_per_class, len(group)), replace=False)
        sample = group.iloc[sample_idx]
        for _, row in tqdm(
            sample.iterrows(), total=len(sample), desc=f"sample {cls}", leave=False
        ):
            full = data_root.parent / row["path"]
            with Image.open(full) as img:
                arr = np.asarray(img)
                rows.append(
                    {
                        "class_4": cls,
                        "subject": row["subject"],
                        "slice_idx": row["slice_idx"],
                        "width": img.width,
                        "height": img.height,
                        "mode": img.mode,
                        "ndim": arr.ndim,
                        "dtype": str(arr.dtype),
                        "intensity_min": int(arr.min()),
                        "intensity_max": int(arr.max()),
                        "intensity_mean": float(arr.mean()),
                        "intensity_std": float(arr.std()),
                        "fraction_zero": float((arr == 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Visualizações
# ---------------------------------------------------------------------------

def fig_class_distribution(df: pd.DataFrame, paths: Paths) -> None:
    summary = class_summary(df, "class_4")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    summary["n_subjects"].plot(kind="bar", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Pacientes únicos por classe (4 classes)")
    axes[0].set_ylabel("nº pacientes")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(summary["n_subjects"].values):
        axes[0].text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9)

    summary["n_slices"].plot(kind="bar", ax=axes[1], color="#DD8452")
    axes[1].set_title("Slices por classe (4 classes)")
    axes[1].set_ylabel("nº slices")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(summary["n_slices"].values):
        axes[1].text(i, v, f"{int(v):,}".replace(",", "."), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(paths.figures / "class_distribution.png", dpi=130)
    plt.close(fig)


def fig_three_class_comparison(df: pd.DataFrame, paths: Paths) -> None:
    summary_3 = class_summary(df, "class_3")
    summary_b = class_summary(df, "class_binary")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    summary_3[["n_subjects", "n_slices"]].plot(
        kind="bar", ax=axes[0], color=["#4C72B0", "#DD8452"], secondary_y="n_slices"
    )
    axes[0].set_title("Esquema 3 classes (Mild+Moderate fundidos)")
    axes[0].tick_params(axis="x", rotation=15)

    summary_b[["n_subjects", "n_slices"]].plot(
        kind="bar", ax=axes[1], color=["#4C72B0", "#DD8452"], secondary_y="n_slices"
    )
    axes[1].set_title("Esquema binário (Demented vs Non)")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(paths.figures / "class_distribution_3_and_binary.png", dpi=130)
    plt.close(fig)


def fig_slices_per_subject(df: pd.DataFrame, paths: Paths) -> None:
    per_subj = df.groupby(["class_4", "subject"]).size().rename("n_slices").reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=per_subj, x="class_4", y="n_slices", ax=ax, color="#4C72B0")
    sns.stripplot(data=per_subj, x="class_4", y="n_slices", ax=ax, color="black", size=2.5, alpha=0.5)
    ax.set_title("Slices por paciente (variação reflete nº de aquisições mpr)")
    ax.set_xlabel("classe")
    ax.set_ylabel("slices por paciente")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(paths.figures / "slices_per_subject.png", dpi=130)
    plt.close(fig)


def fig_slice_idx_distribution(df: pd.DataFrame, paths: Paths) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for cls, color in zip(
        ["non_demented", "very_mild", "mild", "moderate"],
        ["#4C72B0", "#55A868", "#DD8452", "#C44E52"],
    ):
        sub = df[df["class_4"] == cls]
        ax.hist(sub["slice_idx"], bins=40, alpha=0.55, label=cls, color=color)
    ax.set_title("Distribuição do índice da fatia axial (deve ser ~uniforme entre classes)")
    ax.set_xlabel("slice_idx")
    ax.set_ylabel("contagem")
    ax.legend()
    plt.tight_layout()
    plt.savefig(paths.figures / "slice_idx_distribution.png", dpi=130)
    plt.close(fig)


def fig_example_grid(df: pd.DataFrame, data_root: Path, paths: Paths) -> None:
    rng = np.random.default_rng(seed=7)
    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    for row, cls in enumerate(["non_demented", "very_mild", "mild", "moderate"]):
        sub = df[df["class_4"] == cls]
        # pega 4 sujeitos diferentes, fatia central (~130) quando possível
        subjects = sub["subject"].unique()
        chosen = rng.choice(subjects, size=min(4, len(subjects)), replace=False)
        for col, subj in enumerate(chosen):
            subj_slices = sub[sub["subject"] == subj]
            mid = subj_slices.iloc[len(subj_slices) // 2]
            full = data_root.parent / mid["path"]
            with Image.open(full) as img:
                axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_title(f"{cls}\n{subj} slice {mid['slice_idx']}", fontsize=8)
            axes[row, col].axis("off")
        # se houver menos de 4 sujeitos, esconde os eixos restantes
        for col in range(len(chosen), 4):
            axes[row, col].axis("off")
    plt.suptitle("Exemplos: 1 fatia central por paciente (4 pacientes por classe)", fontsize=12)
    plt.tight_layout()
    plt.savefig(paths.figures / "examples_grid.png", dpi=130)
    plt.close(fig)


def fig_intensity_histogram(img_stats: pd.DataFrame, paths: Paths) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=img_stats, x="class_4", y="intensity_mean", ax=ax)
    ax.set_title("Intensidade média dos pixels (amostra por classe)")
    ax.set_xlabel("classe")
    ax.set_ylabel("média de intensidade (0-255)")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    plt.savefig(paths.figures / "intensity_mean_by_class.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Relatório markdown
# ---------------------------------------------------------------------------

def write_report(
    df: pd.DataFrame,
    sanity: dict,
    img_stats: pd.DataFrame,
    paths: Paths,
) -> None:
    s4 = class_summary(df, "class_4")
    s3 = class_summary(df, "class_3")
    sb = class_summary(df, "class_binary")

    lines: list[str] = []
    lines.append("# EDA — OASIS-1 (versão Kaggle pré-processada)\n")
    lines.append(f"- Total de slices: **{len(df):,}**".replace(",", "."))
    lines.append(f"- Total de pacientes únicos: **{df['subject'].nunique()}**")
    lines.append(f"- Aquisições únicas (subject × session × mpr): **{df[['subject','session','mpr']].drop_duplicates().shape[0]}**")
    lines.append(f"- Slice indices observados: {sanity['slice_idx_range_global'][0]}–{sanity['slice_idx_range_global'][1]}")
    lines.append(f"- mpr observado: {sanity['mpr_unique']}")
    lines.append(f"- Sessões observadas: MR {sanity['unique_sessions']}\n")

    lines.append("## Sanity checks")
    leaks = sanity["subjects_with_multiple_labels"]
    if leaks:
        lines.append(f"- ⚠️ **{len(leaks)} sujeitos com múltiplas classes** — investigar antes do split.")
        for subj, n in leaks.items():
            lines.append(f"  - `{subj}`: {n} classes distintas")
    else:
        lines.append("- ✅ Nenhum sujeito aparece em mais de uma classe.\n")

    lines.append("\n## Distribuição (4 classes — original do Kaggle)\n")
    lines.append(s4.to_markdown())
    lines.append("\n## Distribuição (3 classes — Mild+Moderate fundidos)\n")
    lines.append(s3.to_markdown())
    lines.append("\n## Distribuição (binário — fallback do plano original)\n")
    lines.append(sb.to_markdown())

    if not img_stats.empty:
        lines.append("\n## Estatísticas das imagens (amostra)\n")
        agg = (
            img_stats.groupby("class_4")
            .agg(
                width=("width", "first"),
                height=("height", "first"),
                mode=("mode", "first"),
                intensity_mean=("intensity_mean", "mean"),
                intensity_std=("intensity_std", "mean"),
                fraction_zero=("fraction_zero", "mean"),
            )
            .round(2)
        )
        lines.append(agg.to_markdown())

    lines.append("\n## Figuras geradas")
    for p in sorted(paths.figures.glob("*.png")):
        rel = p.relative_to(paths.out_dir)
        lines.append(f"- `{rel.as_posix()}`")

    (paths.out_dir / "eda_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. Entrypoint
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/eda"))
    parser.add_argument("--sample-size", type=int, default=30, help="Imagens amostradas por classe para checagem.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.data_root.exists():
        print(f"[ERRO] data-root não existe: {args.data_root.resolve()}", file=sys.stderr)
        return 2

    paths = Paths.build(args.data_root, args.out_dir)

    print(f"== Construindo manifesto a partir de {args.data_root} ...")
    df = build_manifest(args.data_root)
    print(f"   {len(df):,} slices indexados.".replace(",", "."))

    df.to_csv(paths.out_dir / "manifest.csv", index=False)
    print(f"   manifest.csv salvo em {paths.out_dir}")
    n_acq_global = df[["subject", "session", "mpr"]].drop_duplicates().shape[0]
    print(f"   {n_acq_global} aquisicoes unicas (subject x session x mpr).")

    print("== Sanity checks ...")
    sanity = sanity_checks(df)

    print("== Sumários por classe ...")
    s4 = class_summary(df, "class_4")
    s3 = class_summary(df, "class_3")
    sb = class_summary(df, "class_binary")
    print("\n[4 classes]\n", s4.to_string())
    print("\n[3 classes]\n", s3.to_string())
    print("\n[binário]\n", sb.to_string())

    print("\n== Amostragem de imagens (resolução, modo, intensidade) ...")
    img_stats = sample_image_stats(df, args.data_root, n_per_class=args.sample_size)
    img_stats.to_csv(paths.out_dir / "image_sample_stats.csv", index=False)

    print("== Gerando figuras ...")
    fig_class_distribution(df, paths)
    fig_three_class_comparison(df, paths)
    fig_slices_per_subject(df, paths)
    fig_slice_idx_distribution(df, paths)
    fig_example_grid(df, args.data_root, paths)
    fig_intensity_histogram(img_stats, paths)

    print("== Salvando summary.json e report.md ...")
    summary = {
        "n_slices": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "n_acquisitions": int(df[["subject", "session", "mpr"]].drop_duplicates().shape[0]),
        "sanity": sanity,
        "class_4": s4.reset_index().to_dict(orient="records"),
        "class_3": s3.reset_index().to_dict(orient="records"),
        "class_binary": sb.reset_index().to_dict(orient="records"),
    }
    (paths.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(df, sanity, img_stats, paths)

    print(f"\nOK. Saídas em {paths.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
