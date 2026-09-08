"""M0 smoke test: the package imports and the repo invariants hold.

Replaced by real coverage as each milestone lands; the seeding invariant below
stays permanently, since `make sim` reproducibility depends on it.
"""

import pathlib
import re

import expkit

REPO = pathlib.Path(__file__).resolve().parents[1]


def test_package_imports():
    assert expkit.__version__


def test_no_legacy_global_numpy_rng():
    """Every study must use numpy.random.default_rng(seed).

    The legacy global API (np.random.seed / np.random.normal / ...) shares one
    process-wide state, so adding or reordering a study silently changes every
    downstream draw and results/ stops being reproducible.
    """
    offenders = []
    # Trailing [a-z_] keeps prose mentions of "numpy.random.*" in docstrings from tripping this.
    legacy = re.compile(r"\bn(?:p|umpy)\.random\.(?!default_rng)[a-z_]")
    for path in list((REPO / "src").rglob("*.py")) + list((REPO / "sims").rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if legacy.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}")
    assert not offenders, f"legacy global numpy RNG used at: {offenders}"
