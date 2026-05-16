"""The TARGET agent under test: a refund-policy adjudicator.

This is the real LLM agent Telltale regression-tests. It is intentionally a
plausible enterprise agent (the kind a UiPath customer ships): given a refund
ticket it returns a structured decision. Versioned by AgentConfig so a change
(prompt / model / temperature) produces a distinct, reproducible "candidate".

Real Gemini calls via telltale.llm — no mock path.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from . import llm

ROOT = Path(__file__).resolve().parents[1]

BASELINE_SYSTEM = (
    "You are a refund-policy adjudicator for an electronics retailer.\n"
    "Policy: refunds allowed within 30 days with receipt. Items 31-60 days get "
    "store credit only. Beyond 60 days: deny. Opened software: deny. Physical "
    "damage by customer: deny. Manager override flag forces approval of the full "
    "requested amount. Never refund more than the requested amount.\n"
    "Respond ONLY with a JSON object: "
    '{"decision":"approve|deny","amount":<number>,"reason":"<one sentence>"}'
)


@dataclasses.dataclass(frozen=True)
class AgentConfig:
    version: str
    system: str = BASELINE_SYSTEM
    model: str = "gemini-2.5-pro"
    temperature: float = 0.7

    @staticmethod
    def baseline() -> "AgentConfig":
        return AgentConfig(version="baseline")

    @staticmethod
    def load(path: str) -> "AgentConfig":
        d = json.loads(Path(path).read_text())
        return AgentConfig(**d)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=2) + "\n")


def adjudicate(cfg: AgentConfig, ticket: dict) -> dict:
    """One real adjudication. Returns {decision, amount, reason}. No fallback."""
    user = (
        f"Ticket #{ticket['id']}\n"
        f"Item: {ticket['item']}\nRequested refund: ${ticket['requested']:.2f}\n"
        f"Days since purchase: {ticket['days']}\n"
        f"Has receipt: {ticket['receipt']}\n"
        f"Condition: {ticket['condition']}\n"
        f"Manager override flag: {ticket['manager_override']}\n"
        "Adjudicate per policy."
    )
    raw = llm.generate(cfg.system, user, model=cfg.model, temperature=cfg.temperature)
    obj = llm.extract_json(raw)
    return {
        "decision": str(obj.get("decision", "")).strip().lower(),
        "amount": float(obj.get("amount", 0) or 0),
        "reason": str(obj.get("reason", "")).strip(),
        "_raw": raw,
    }
