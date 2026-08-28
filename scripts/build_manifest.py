from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "MANIFEST.sha256"
EXCLUDED_PARTS = {".git", ".venv", "__pycache__", "outputs", "results"}


def included_files():
    for path in sorted(ROOT.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or path == OUTPUT:
            continue
        if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        yield path


def main() -> None:
    rows = []
    for path in included_files():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
