from __future__ import annotations

import compileall
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".ps1", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".example"}
REQUIRED = [
    ROOT / "README.md",
    ROOT / "requirements.txt",
    ROOT / "docs" / "MANUSCRIPT_ALIGNMENT.md",
    ROOT / "docs" / "REPRODUCIBILITY_STATUS.md",
    ROOT / "data" / "README.md",
    ROOT / "configs" / "README.md",
    ROOT / "configs" / "task_spec.classification.example.json",
    ROOT / "autologic" / "utils" / "task_protocol.py",
    ROOT / "scripts" / "build_manifest.py",
    ROOT / "autologic" / "ensemble" / "classification_ensemble" / "classification_auto_ensemble.py",
    ROOT / "autologic" / "ensemble" / "regression_ensemble" / "regression_auto_ensemble.py",
    ROOT / "autologic" / "ensemble" / "cluster_ensemble" / "cluster_auto_ensemble_hidden_evaluator.py",
    ROOT / "reproduce" / "fig4" / "run_full_benchmark.ps1",
]

PATH_PATTERNS = {
    "Windows absolute path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s`\"']+"),
    "Unix home path": re.compile(r"/(?:home|root)/[^\s`\"']+"),
}
SECRET_PATTERNS = {
    "OpenAI-style key": re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{12,}"),
    "private key block": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
}


def iter_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == Path(__file__).resolve():
            continue
        if any(part in {".git", ".venv", "__pycache__", "outputs", "results"} for part in path.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name == ".gitignore":
            yield path


def main() -> int:
    errors: list[str] = []
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        errors.append("Missing required files: " + ", ".join(missing))

    if not compileall.compile_dir(ROOT, quiet=1, force=True):
        errors.append("Python compilation failed.")

    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if tests.returncode != 0:
        errors.append("Unit tests failed:\n" + (tests.stdout + tests.stderr).strip())

    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in {**PATH_PATTERNS, **SECRET_PATTERNS}.items():
            match = pattern.search(text)
            if match:
                errors.append(f"{label} in {path.relative_to(ROOT)}: {match.group(0)[:80]}")

    prohibited = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pkl", ".pickle", ".pt", ".pth", ".ckpt"}
    ]
    if prohibited:
        errors.append("Unexpected binary data/model files: " + ", ".join(map(str, prohibited)))

    if errors:
        print("Release checks failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Release checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
