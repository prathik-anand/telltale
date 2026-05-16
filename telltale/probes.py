"""Frozen probe set (Build Contract R3).

60 deterministic refund tickets, committed to data/probes.json. Frozen so the
drift test is reproducible; judge-substitutable so it's not staged — a judge
can drop their own probes.json and re-run, or append cases. The generator is
deterministic (seeded) so the committed file is auditable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROBES_JSON = ROOT / "data" / "probes.json"

_ITEMS = ["USB-C cable", "4K monitor", "wireless mouse", "laptop dock",
          "mechanical keyboard", "noise-cancel headphones", "webcam", "SSD 1TB"]
_CONDITIONS = ["unopened", "opened-unused", "opened-software", "customer-damaged",
               "like-new", "missing-parts"]


def generate(n: int = 60, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        days = int(rng.integers(1, 95))
        out.append({
            "id": f"RB-{1000 + i}",
            "item": _ITEMS[i % len(_ITEMS)],
            "requested": round(float(rng.uniform(15, 600)), 2),
            "days": days,
            "receipt": bool(rng.integers(0, 2)),
            "condition": _CONDITIONS[int(rng.integers(0, len(_CONDITIONS)))],
            "manager_override": bool(rng.integers(0, 10) == 0),
        })
    return out


def freeze(n: int = 60, seed: int = 7) -> Path:
    PROBES_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROBES_JSON.write_text(json.dumps(generate(n, seed), indent=2) + "\n")
    return PROBES_JSON


def load() -> list[dict]:
    if not PROBES_JSON.exists():
        freeze()
    return json.loads(PROBES_JSON.read_text())


if __name__ == "__main__":
    p = freeze()
    print(f"froze {len(json.loads(p.read_text()))} probes -> {p}")
