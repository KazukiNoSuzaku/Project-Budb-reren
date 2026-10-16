"""Enforce the library policy mechanically, so it cannot silently regress.

scikit-learn (and SciPy, which it pulls in transitively) is installed, so the
only real guarantee that we do not use it is to check the source.
"""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "budbaereren"

FORBIDDEN = {"torch", "tensorflow", "jax", "keras", "transformers", "scipy", "autograd"}
# scikit-learn is allowed in exactly one module: the dataset loader.
SKLEARN_ALLOWED = {PACKAGE / "train" / "data.py"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_forbidden_imports():
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        roots = _imported_roots(path)
        if roots & FORBIDDEN:
            offenders.append(f"{path}: {sorted(roots & FORBIDDEN)}")
        if "sklearn" in roots and path not in SKLEARN_ALLOWED:
            offenders.append(f"{path}: sklearn outside the dataset loader")
    assert not offenders, "\n".join(offenders)
