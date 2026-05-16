"""Programmatic real drift run for the web UI.

Reuses the EXACT real code path as the CLI (telltale.run internals):
real Gemini generations -> real Gemini embeddings -> the pre-registered
Mann-Whitney + Benjamini-Hochberg + effect-size engine. No mock, fail-closed.

Two entrypoints the web app calls:
  live_run(candidate_path, n_probes, k) -> real verdict dict (small, capturable)
  load_evidence()                       -> the committed real headline runs
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from . import probes
from .agent import AgentConfig
from .config import MissingCredential, prereg_path
from .drift import PreReg, run_drift, negative_control_ok
from .run import _gen_outputs, _probe_metric  # the SAME real primitives the CLI uses

ROOT = Path(__file__).resolve().parents[1]
EVID = ROOT / "evidence"


def live_run(candidate_path: str, n_probes: int = 4, k: int = 2) -> dict:
    """Run a REAL (small, capturable) drift job. Raises MissingCredential with
    no key — never returns a fake verdict (F6)."""
    pr_raw = yaml.safe_load(open(prereg_path()))
    embed_model = pr_raw["embed_model"]
    prereg = PreReg.from_yaml(prereg_path())
    base = AgentConfig.baseline()
    cand = AgentConfig.load(candidate_path)
    pset = probes.load()[:n_probes]

    per_probe, control = {}, {}
    for t in pset:
        b1 = _gen_outputs(base, t, k)
        b2 = _gen_outputs(base, t, k)            # negative control draw
        cd = _gen_outputs(cand, t, k)
        bs, cs = _probe_metric(b1, cd, embed_model)
        _, ctrl = _probe_metric(b1, b2, embed_model)
        per_probe[t["id"]] = (bs, cs)
        control[t["id"]] = (bs, ctrl)

    verdicts = run_drift(per_probe, prereg)
    ctrl_v = run_drift(control, prereg)
    regressed = [v.as_dict() for v in verdicts if v.regressed]
    nc_ok = negative_control_ok(ctrl_v, prereg)
    return {
        "live": True,
        "candidate": cand.version,
        "model": cand.model,
        "probes": len(pset),
        "k_runs": k,
        "prereg": {"alpha": prereg.alpha, "delta": prereg.delta},
        "negative_control_passed": nc_ok,
        "regressed_count": len(regressed),
        "regressed": regressed[:6],
        "verdict": ("REGRESSION_DETECTED" if regressed and nc_ok
                    else "CONTROL_FAILED" if not nc_ok else "NO_REGRESSION"),
    }


def load_evidence() -> dict:
    """The committed REAL headline runs (the full 24xK4 results on disk)."""
    out = {}
    for key, fn in (("model_downgrade", "verdict_modeldowngrade_REGRESSION.json"),
                     ("prompt_tweak", "verdict_promptchange_NOREG.json"),
                     ("handtests_on_regressed", "handtests_pass_on_regressed_model.json")):
        p = EVID / fn
        out[key] = json.loads(p.read_text()) if p.exists() else None
    return out
