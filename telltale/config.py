"""Config + credential loading. Fail-closed: no key => hard error, never a mock."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class MissingCredential(RuntimeError):
    """Raised when GEMINI_API_KEY is absent. We do NOT fall back to a stub —
    a faked model call would violate Build Contract R1/R3 (F6)."""


def load_env() -> None:
    """Load project .env (KEY=VALUE lines) into os.environ if not already set."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


def gemini_key() -> str:
    load_env()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise MissingCredential(
            "GEMINI_API_KEY not set. Fill 2026-05-16/uipath-agenthack/project/.env "
            "with GEMINI_API_KEY=<your key>. Telltale will not mock the model — "
            "the demo is real or it does not run (Build Contract R1/R3)."
        )
    return key


def prereg_path() -> str:
    return str(ROOT / "prereg.yaml")
