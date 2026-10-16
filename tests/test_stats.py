import numpy as np
import pytest

from budbaereren.train.stats import paired_comparison, t_critical_975


def test_paired_t_against_hand_computation():
    # d = [1, 2, 3, 4, 5]: mean 3, sd sqrt(2.5), se sqrt(0.5), t = 3/sqrt(0.5) = 4.2426
    r = paired_comparison([2, 4, 6, 8, 10], [1, 2, 3, 4, 5])
    assert r.mean_diff == pytest.approx(3.0)
    assert r.sd_diff == pytest.approx(np.sqrt(2.5))
    assert r.t_stat == pytest.approx(3 / np.sqrt(0.5))
    assert r.ci_low == pytest.approx(3 - 2.776 * np.sqrt(0.5))
    assert r.ci_high == pytest.approx(3 + 2.776 * np.sqrt(0.5))
    assert r.real and r.wins == 5


def test_noise_is_not_called_real():
    r = paired_comparison([0.98, 0.97, 0.99, 0.98, 0.97], [0.97, 0.98, 0.98, 0.99, 0.97])
    assert not r.real and r.ci_low < 0 < r.ci_high


def test_pairing_removes_a_shared_seed_effect():
    seed_effect = np.array([0.0, 0.05, -0.04, 0.03, -0.02])  # large between-seed variance
    a = 0.95 + seed_effect + 0.002
    b = 0.95 + seed_effect + np.array([0.0, 0.0005, -0.0005, 0.0003, -0.0003])
    assert paired_comparison(a, b).real  # unpaired, the 0.002 gap would drown in 0.035 sd


def test_t_table():
    assert t_critical_975(4) == 2.776 and t_critical_975(1) == 12.706
    with pytest.raises(ValueError):
        t_critical_975(0)
