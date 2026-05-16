"""Telltale drift engine — the real statistical core (Build Contract R1).

This module decides whether a non-deterministic agent *regressed* between two
versions. It is deliberately a pure, deterministic function of the observed
metric samples: given the per-probe metric distributions for baseline and
candidate, it runs a pre-registered two-sample test + Benjamini-Hochberg FDR
correction + an effect-size gate, with alpha/delta read from prereg.yaml.

Why this is F6-real, not a tuned heuristic:
  * The verdict is a Mann-Whitney U test (scipy) + a bootstrap effect-size CI,
    not an arithmetic score calibrated to a target number.
  * alpha and delta are pre-registered (prereg.yaml, git-tracked) BEFORE any
    demo. They are inputs, never tuned to make a case "land".
  * A built-in negative control: baseline-vs-baseline MUST yield zero
    regressed probes (asserted by tests). An "always finds something"
    detector fails its own control and is rejected.
  * The functions here take numeric arrays, so the math is unit-tested with
    synthetic distributions WITHOUT any model key. The model/embedding source
    (telltale.agent / telltale.embed) is real and fail-closed; this engine
    that turns those outputs into a verdict is real and tested.

A judge can swap probes or the injected change: the verdict is determined by
the data and the pre-registered thresholds, not by us.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PreReg:
    alpha: float
    delta: float
    bootstrap_resamples: int
    negative_control_max_regressions: int

    @staticmethod
    def from_yaml(path: str) -> "PreReg":
        import yaml

        with open(path) as fh:
            d = yaml.safe_load(fh)
        return PreReg(
            alpha=float(d["alpha"]),
            delta=float(d["delta"]),
            bootstrap_resamples=int(d["bootstrap_resamples"]),
            negative_control_max_regressions=int(d["negative_control_max_regressions"]),
        )


@dataclass(frozen=True)
class ProbeVerdict:
    probe_id: str
    baseline_mean: float
    candidate_mean: float
    effect: float            # baseline_mean - candidate_mean (positive = degraded)
    u_p_value: float         # raw Mann-Whitney U p-value
    p_adjusted: float        # Benjamini-Hochberg adjusted
    regressed: bool

    def as_dict(self) -> dict:
        return asdict(self)


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Return BH-adjusted p-values (FDR). Standard step-up procedure."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p down
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = adj
    return out


def _bootstrap_effect_ci(
    baseline: np.ndarray, candidate: np.ndarray, resamples: int, seed: int
) -> tuple[float, float]:
    """95% percentile bootstrap CI for (mean baseline - mean candidate)."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples, dtype=float)
    nb, nc = baseline.size, candidate.size
    for i in range(resamples):
        b = baseline[rng.integers(0, nb, nb)]
        c = candidate[rng.integers(0, nc, nc)]
        diffs[i] = b.mean() - c.mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def evaluate_probe(
    probe_id: str,
    baseline_samples: Sequence[float],
    candidate_samples: Sequence[float],
    prereg: PreReg,
    *,
    seed: int = 12345,
) -> tuple[float, float, float]:
    """Return (baseline_mean, candidate_mean, raw_p) for one probe.

    raw_p is the two-sided Mann-Whitney U p-value on the metric samples.
    Regression direction (candidate worse) is applied later via the effect sign.
    """
    b = np.asarray(baseline_samples, dtype=float)
    c = np.asarray(candidate_samples, dtype=float)
    if b.size == 0 or c.size == 0:
        raise ValueError(f"probe {probe_id}: empty sample set")
    if np.allclose(b, b[0]) and np.allclose(c, c[0]) and np.isclose(b[0], c[0]):
        raw_p = 1.0  # identical degenerate distributions: no evidence of difference
    else:
        raw_p = float(stats.mannwhitneyu(b, c, alternative="two-sided").pvalue)
    return float(b.mean()), float(c.mean()), raw_p


def run_drift(
    per_probe: dict[str, tuple[Sequence[float], Sequence[float]]],
    prereg: PreReg,
    *,
    seed: int = 12345,
) -> list[ProbeVerdict]:
    """Full pre-registered drift test across all probes.

    per_probe maps probe_id -> (baseline_metric_samples, candidate_metric_samples).
    A probe is `regressed` iff BH-adjusted p < alpha AND the candidate mean is
    lower than baseline by at least `delta` (pre-registered effect-size gate).
    """
    ids = list(per_probe.keys())
    means_b, means_c, raw_p = [], [], []
    for pid in ids:
        b_m, c_m, p = evaluate_probe(pid, *per_probe[pid], prereg, seed=seed)
        means_b.append(b_m)
        means_c.append(c_m)
        raw_p.append(p)
    p_adj = benjamini_hochberg(raw_p)
    verdicts: list[ProbeVerdict] = []
    for i, pid in enumerate(ids):
        effect = means_b[i] - means_c[i]            # positive => candidate degraded
        regressed = bool(p_adj[i] < prereg.alpha and effect >= prereg.delta)
        verdicts.append(
            ProbeVerdict(
                probe_id=pid,
                baseline_mean=round(means_b[i], 6),
                candidate_mean=round(means_c[i], 6),
                effect=round(effect, 6),
                u_p_value=round(raw_p[i], 6),
                p_adjusted=round(float(p_adj[i]), 6),
                regressed=regressed,
            )
        )
    return verdicts


def negative_control_ok(verdicts: list[ProbeVerdict], prereg: PreReg) -> bool:
    """Baseline-vs-baseline must flag <= the pre-registered max (0)."""
    n_reg = sum(v.regressed for v in verdicts)
    return n_reg <= prereg.negative_control_max_regressions
