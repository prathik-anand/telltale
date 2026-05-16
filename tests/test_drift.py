"""Deterministic proof that the drift engine is REAL (no model key needed).

These tests exercise telltale.drift with synthetic numeric distributions. They
test the *statistics*, not the product wow — the wow's data source
(real Gemini outputs -> real embeddings) is real and fail-closed elsewhere.
This is the honest distinction the Build Contract R1 requires: the verdict
machinery is a genuine pre-registered hypothesis test, provable without creds.
"""
import numpy as np
import pytest

from telltale.drift import (
    PreReg,
    benjamini_hochberg,
    run_drift,
    negative_control_ok,
)

PR = PreReg(alpha=0.05, delta=0.04, bootstrap_resamples=500,
            negative_control_max_regressions=0)


def _samples(mean, n=5, sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return list(np.clip(rng.normal(mean, sd, n), 0, 1))


def test_negative_control_identical_distributions_flag_zero():
    # baseline vs an independent draw from the SAME distribution -> no regression
    per_probe = {
        f"p{i}": (_samples(0.90, seed=i), _samples(0.90, seed=i + 1000))
        for i in range(20)
    }
    v = run_drift(per_probe, PR)
    assert negative_control_ok(v, PR)
    assert sum(x.regressed for x in v) == 0


def test_real_regression_is_detected():
    # candidate clearly degraded (0.90 -> 0.78) across many probes -> flagged
    per_probe = {
        f"p{i}": (_samples(0.90, n=8, seed=i), _samples(0.78, n=8, seed=i + 500))
        for i in range(15)
    }
    v = run_drift(per_probe, PR)
    assert sum(x.regressed for x in v) >= 12  # the bulk must trip


def test_effect_size_gate_blocks_trivial_shift():
    # statistically detectable but tiny (0.900 -> 0.892, below delta=0.04) -> NOT a regression
    per_probe = {
        f"p{i}": (_samples(0.900, n=40, sd=0.002, seed=i),
                  _samples(0.892, n=40, sd=0.002, seed=i + 700))
        for i in range(10)
    }
    v = run_drift(per_probe, PR)
    assert sum(x.regressed for x in v) == 0  # effect < delta -> the gate holds


def test_improvement_is_not_a_regression():
    # candidate BETTER (0.85 -> 0.95): effect is negative -> never "regressed"
    per_probe = {
        f"p{i}": (_samples(0.85, n=8, seed=i), _samples(0.95, n=8, seed=i + 300))
        for i in range(10)
    }
    v = run_drift(per_probe, PR)
    assert sum(x.regressed for x in v) == 0


def test_benjamini_hochberg_correct_and_monotone():
    p = [0.001, 0.008, 0.039, 0.041, 0.9]
    adj = benjamini_hochberg(p)
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-9)  # monotone in p-order
    assert np.all((adj >= 0) & (adj <= 1))
    assert adj[0] < adj[-1]


def test_prereg_thresholds_are_inputs_not_hardcoded():
    # Loosening delta turns a sub-threshold shift into a flagged regression:
    # proves the verdict is governed by the pre-registered file, not a constant.
    per_probe = {
        f"p{i}": (_samples(0.90, n=30, sd=0.002, seed=i),
                  _samples(0.875, n=30, sd=0.002, seed=i + 900))
        for i in range(10)
    }
    strict = run_drift(per_probe, PreReg(0.05, 0.04, 500, 0))
    loose = run_drift(per_probe, PreReg(0.05, 0.01, 500, 0))
    assert sum(x.regressed for x in strict) == 0
    assert sum(x.regressed for x in loose) >= 8


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
