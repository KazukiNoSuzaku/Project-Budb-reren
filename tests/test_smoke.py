import numpy as np

import budbaereren
from budbaereren.utils import seed_everything


def test_package_imports():
    assert budbaereren.__version__


def test_seed_everything_is_deterministic():
    a = seed_everything(123).standard_normal(5)
    b = seed_everything(123).standard_normal(5)
    np.testing.assert_array_equal(a, b)
