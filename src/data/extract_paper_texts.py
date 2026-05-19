"""Extrai texto de todos os PDFs em docs/Papers/Trabalhos Relacionados/ para
arquivos .txt UTF-8 numa pasta temporária, para auditoria de metodologia."""
import sys
from pathlib import Path

from pypdf import PdfReader

PAPERS = Path("docs/Papers/Trabalhos Relacionados")
OUT = Path("_paper_texts")
OUT.mkdir(exist_ok=True)

for pdf in sorted(PAPERS.glob("*.pdf")):
    try:
        reader = PdfReader(pdf)
        parts = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            parts.append(f"\n===== PAGE {i+1} =====\n{txt}")
        # nome de saída seguro
        safe = pdf.stem.replace(" ", "_").replace("'", "").replace("’", "")[:60]
        out_file = OUT / f"{safe}.txt"
        out_file.write_text("\n".join(parts), encoding="utf-8")
        print(f"OK  {len(reader.pages):3d}p  {out_file.name}")
    except Exception as e:  # noqa: BLE001
        print(f"ERRO em {pdf.name}: {e}")
