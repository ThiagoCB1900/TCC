"""Varredura sistemática nos PDFs em docs/Papers/Trabalhos Relacionados/
para descobrir como cada paper trata o desbalanceamento de classes.

Imprime, para cada paper, os trechos contendo qualquer termo da lista
de busca, com contexto de ~200 caracteres. Útil para fundamentar ADR-0007.

Uso:
    python -m src.data.survey_imbalance_handling

Saída:
    stdout — trechos relevantes por paper
    docs/findings/_imbalance_survey_excerpts.md — mesma saída em markdown
"""
from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

PAPERS_DIR = Path("docs/Papers/Trabalhos Relacionados")
OUT_MD = Path("docs/findings/_imbalance_survey_excerpts.md")

# Termos buscados — variações em inglês cobrindo as principais técnicas
SEARCH_TERMS = [
    r"class[- ]?imbalance",
    r"imbalanced",
    r"balanced[- ]?accuracy",
    r"weighted[- ]?cross[- ]?entropy",
    r"weighted[- ]?loss",
    r"class[- ]?weight",
    r"focal[- ]?loss",
    r"oversampl",
    r"undersampl",
    r"SMOTE",
    r"WeightedRandomSampler",
    r"weighted[- ]?sampl",
    r"random[- ]?sampler",
    r"data[- ]?augmentation",
    r"augmentation",
    r"upsampling",
    r"downsampling",
    r"resampling",
    r"minority class",
    r"majority class",
    r"stratif",
]

PATTERN = re.compile("|".join(SEARCH_TERMS), re.IGNORECASE)


def extract_text(pdf_path: Path) -> str:
    """Extrai todo o texto do PDF, juntando páginas. Robusto a falhas em páginas individuais."""
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:  # noqa: BLE001 — pypdf pode lançar várias exceções
        return f"[ERRO ao abrir: {e}]"
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            parts.append(page.extract_text() or "")
        except Exception as e:  # noqa: BLE001
            parts.append(f"[ERRO p.{i}: {e}]")
    return "\n".join(parts)


def find_excerpts(text: str, context_chars: int = 220) -> list[tuple[str, str]]:
    """Encontra todas as ocorrências dos termos e retorna (termo, trecho com contexto)."""
    excerpts: list[tuple[str, str]] = []
    for match in PATTERN.finditer(text):
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)
        snippet = text[start:end].replace("\n", " ").strip()
        snippet = re.sub(r"\s+", " ", snippet)
        excerpts.append((match.group(0).lower(), snippet))
    return excerpts


def deduplicate(excerpts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove trechos quase-duplicados (mesma frase pega por termos diferentes)."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for term, snip in excerpts:
        # chave: 80 chars no meio do snippet
        mid = snip[len(snip) // 2 - 40 : len(snip) // 2 + 40]
        key = re.sub(r"\W+", "", mid).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((term, snip))
    return out


def safe_print(s: str) -> None:
    """Imprime fallback ASCII-safe para evitar erro de encoding em Windows cp1252."""
    import sys

    try:
        print(s)
    except UnicodeEncodeError:
        sys.stdout.write(s.encode("ascii", errors="replace").decode("ascii") + "\n")


def main() -> int:
    if not PAPERS_DIR.exists():
        safe_print(f"[ERRO] pasta nao encontrada: {PAPERS_DIR.resolve()}")
        return 2

    pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
    safe_print(f"Auditando {len(pdfs)} PDFs em {PAPERS_DIR}...\n")

    md_lines: list[str] = [
        "# Excertos sobre tratamento de desbalanceamento de classes",
        "",
        "Geração automática (`src/data/survey_imbalance_handling.py`).",
        "Trechos extraídos com contexto de ~220 caracteres ao redor de termos de busca.",
        "**Não é citável diretamente** — verifique o paper completo antes de usar no TCC.",
        "",
        f"Termos buscados: `{', '.join(SEARCH_TERMS)}`.",
        "",
    ]

    for pdf in pdfs:
        md_lines.append(f"\n## {pdf.name}\n")
        text = extract_text(pdf)
        if text.startswith("[ERRO"):
            safe_print(f"=== {pdf.name} === {text}")
            md_lines.append(f"_{text}_")
            continue
        excerpts = deduplicate(find_excerpts(text))
        if not excerpts:
            safe_print(f"=== {pdf.name} === sem ocorrencias relevantes")
            md_lines.append("_(nenhuma ocorrência dos termos de busca)_")
            continue

        safe_print(f"=== {pdf.name} === {len(excerpts)} excerto(s)")
        for term, snip in excerpts[:30]:  # cap por paper para não inundar
            safe_print(f"  [{term}] ...{snip}...")
            md_lines.append(f"- **[{term}]** …{snip}…")
        md_lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    safe_print(f"\nMarkdown consolidado: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
