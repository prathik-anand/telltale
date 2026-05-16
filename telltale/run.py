"""Telltale orchestrator — the real end-to-end run (Build Contract R1+R3).

For each frozen probe: K real Gemini generations from baseline AND candidate.
Metric per output = cosine similarity of its real Gemini embedding to the
baseline behavioral centroid for that probe. If the candidate drifted, its
outputs sit further from the baseline centroid -> the per-probe metric
distribution shifts -> the pre-registered Mann-Whitney+BH+effect-size test
flags it. A negative control (baseline re-run vs baseline) must flag zero.

Fail-closed: no GEMINI_API_KEY -> MissingCredential, exit 2. Never mocks.
On the real platform path this orchestrator is wrapped as a UiPath Maestro
agentic process with each probe a Test Cloud test run (R2, post-creds).

  python -m telltale.run --candidate candidate_config.json --out runs_out/verdict.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import llm, probes
from .agent import AgentConfig, adjudicate
from .config import MissingCredential, prereg_path
from .drift import PreReg, run_drift, negative_control_ok

ROOT = Path(__file__).resolve().parents[1]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def _gen_outputs(cfg: AgentConfig, ticket: dict, k: int) -> list[str]:
    return [adjudicate(cfg, ticket)["_raw"] for _ in range(k)]


def _probe_metric(baseline_texts, candidate_texts, embed_model):
    """baseline_samples = each baseline output's cos to baseline centroid;
    candidate_samples = each candidate output's cos to the SAME centroid."""
    embs = np.array(llm.embed(baseline_texts + candidate_texts, model=embed_model))
    nb = len(baseline_texts)
    b, c = embs[:nb], embs[nb:]
    centroid = b.mean(axis=0)
    return ([_cos(v, centroid) for v in b], [_cos(v, centroid) for v in c])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", default="runs_out/verdict.json")
    args = ap.parse_args()

    import yaml
    pr_raw = yaml.safe_load(open(prereg_path()))
    K = int(pr_raw["k_runs"])
    embed_model = pr_raw["embed_model"]
    prereg = PreReg.from_yaml(prereg_path())

    base = AgentConfig.baseline()
    cand = AgentConfig.load(args.candidate)
    pset = probes.load()

    try:
        per_probe, control = {}, {}
        for t in pset:
            b1 = _gen_outputs(base, t, K)
            b2 = _gen_outputs(base, t, K)          # 2nd baseline draw -> neg control
            cd = _gen_outputs(cand, t, K)
            bs, cs = _probe_metric(b1, cd, embed_model)
            _, ctrl = _probe_metric(b1, b2, embed_model)
            per_probe[t["id"]] = (bs, cs)
            control[t["id"]] = (bs, ctrl)
    except MissingCredential as e:
        print(f"\nFAIL-CLOSED (no mock): {e}\n", file=sys.stderr)
        return 2

    verdicts = run_drift(per_probe, prereg)
    ctrl_verdicts = run_drift(control, prereg)
    regressed = [v.as_dict() for v in verdicts if v.regressed]
    nc_ok = negative_control_ok(ctrl_verdicts, prereg)

    case = {
        "schema": "telltale.maestro_case/v1",
        "target_baseline": base.version,
        "candidate": cand.version,
        "probes": len(pset),
        "k_runs": K,
        "prereg": {"alpha": prereg.alpha, "delta": prereg.delta},
        "negative_control_passed": nc_ok,
        "regressed_count": len(regressed),
        "regressed_probes": regressed,
        "verdict": ("REGRESSION_DETECTED" if regressed and nc_ok
                    else "CONTROL_FAILED" if not nc_ok
                    else "NO_REGRESSION"),
        "human_gate": "AWAITING_APPROVAL" if regressed and nc_ok else "N/A",
        "note": ("Negative control must pass for the verdict to be trusted. "
                 "alpha/delta are pre-registered in prereg.yaml."),
    }
    outp = ROOT / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(case, indent=2) + "\n")
    print(json.dumps({k: case[k] for k in
                      ("verdict", "regressed_count", "negative_control_passed")},
                     indent=2))
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
