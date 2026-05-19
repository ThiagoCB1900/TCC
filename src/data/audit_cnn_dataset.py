"""Auditoria de identificabilidade de paciente no dataset uraninjo (CNN comparativa).

Pergunta central: é possível agrupar as imagens por sujeito/paciente neste dataset?
Se não for, qualquer trabalho que o use (incluindo o notebook CNN do aluno) é
incapaz de fazer split por paciente — só split por imagem, com risco de leakage
não verificável.

Checa, para cada classe (Original e Augmented):
  - padrões de nomenclatura (esqueleto com dígitos -> '#', prefixos alfabéticos);
  - metadados EXIF (PIL) — algum campo de origem?;
  - dimensões e modo das imagens;
  - duplicatas exatas (hash MD5) dentro de cada classe e ENTRE classes;
  - cópias exatas Original <-> Augmented.

Uso (a partir da raiz do repo):
    python src/data/audit_cnn_dataset.py

Resultado sustenta F-0014. Reexecutável para auditoria independente.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

BASE = Path("CNN/Alzheimers disease dataset/Alzheimer's dataset")
ORIGINAL = BASE / "OriginalDataset"
AUGMENTED = BASE / "AugmentedAlzheimerDataset"
CLASSES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]


def hash_file(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def analyze_names(folder: Path) -> None:
    print("\n== Nomenclatura ==")
    for cls in CLASSES:
        files = sorted(p.name for p in (folder / cls).iterdir() if p.is_file())
        skeletons = Counter(re.sub(r"\d+", "#", f) for f in files)
        prefixes = Counter(re.match(r"^([A-Za-z ]*)", f).group(1).strip() for f in files)
        print(f"  [{cls}] {len(files)} arq")
        print(f"     amostra: {files[:4]} ... {files[-2:]}")
        print(f"     esqueletos: {dict(skeletons)}")
        print(f"     prefixos alfabeticos: {dict(prefixes)}")


def analyze_exif_dims(folder: Path, n: int = 4) -> None:
    print("\n== EXIF + dimensoes (amostra) ==")
    for cls in CLASSES:
        files = sorted(p for p in (folder / cls).iterdir() if p.is_file())[:n]
        for p in files:
            with Image.open(p) as img:
                try:
                    raw = img._getexif()  # type: ignore[attr-defined]
                    exif = {TAGS.get(k, k): v for k, v in raw.items()} if raw else None
                except Exception:
                    exif = None
            print(f"  [{cls}] {p.name:18s} size={img.size} mode={img.mode} exif={exif}")
        break  # uma classe basta para ilustrar dims/EXIF


def analyze_duplicates(folder: Path) -> None:
    print("\n== Duplicatas exatas (MD5) ==")
    all_hashes: dict[str, list[str]] = defaultdict(list)
    for cls in CLASSES:
        local: dict[str, int] = defaultdict(int)
        for p in (folder / cls).iterdir():
            if p.is_file():
                h = hash_file(p)
                local[h] += 1
                all_hashes[h].append(f"{cls}/{p.name}")
        n_total = sum(local.values())
        print(f"  [{cls}] {n_total} arq, {len(local)} hashes unicos, "
              f"{sum(1 for v in local.values() if v > 1)} grupos de duplicata interna")

    cross = {h: v for h, v in all_hashes.items() if len({x.split('/')[0] for x in v}) > 1}
    print(f"  CROSS-CLASSE: {len(cross)} imagens identicas em >1 classe "
          f"(se >0 = mesma imagem rotulada em classes diferentes)")
    for v in list(cross.values())[:5]:
        print(f"     {v}")


def cross_original_augmented(cls: str = "NonDemented") -> None:
    print(f"\n== Cross Original<->Augmented ({cls}) ==")
    orig_h = {hash_file(p) for p in (ORIGINAL / cls).iterdir() if p.is_file()}
    total = match = 0
    for p in (AUGMENTED / cls).iterdir():
        if p.is_file():
            total += 1
            if hash_file(p) in orig_h:
                match += 1
    print(f"  Augmented {cls}: {total} arq; copias byte-a-byte de uma Original: {match}")
    print("  Nota: augmentation (flip/rot/zoom) altera pixels => hash muda; portanto")
    print("  match=0 NAO refuta que Augmented derive das mesmas imagens-fonte; apenas")
    print("  mostra que nao sao copias literais.")


def main() -> int:
    print("### AUDITORIA uraninjo (CNN comparativa) — F-0014 ###")
    print("\n# OriginalDataset")
    analyze_names(ORIGINAL)
    analyze_exif_dims(ORIGINAL)
    analyze_duplicates(ORIGINAL)
    cross_original_augmented("NonDemented")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
