"""The hand-written assertion suite — what a human QA actually writes.

These are schema/policy invariants. The point of the whole demo: they PASS on
both the baseline and a semantically-regressed candidate, because semantic
drift (tone, leniency, reasoning quality) doesn't trip a hard assertion. That
is why deterministic tests lie about non-deterministic agents — and why
Telltale's statistical drift test is needed.
"""
from __future__ import annotations


def check(ticket: dict, output: dict) -> list[str]:
    """Return list of failed-assertion messages ([] == all pass)."""
    fails = []
    if output.get("decision") not in {"approve", "deny"}:
        fails.append("decision not in {approve,deny}")
    amt = output.get("amount", 0)
    if not isinstance(amt, (int, float)) or amt < 0:
        fails.append("amount negative or non-numeric")
    if output.get("decision") == "deny" and amt not in (0, 0.0):
        fails.append("denied but amount != 0")
    if amt > ticket["requested"] + 1e-6:
        fails.append("amount exceeds requested")
    if not str(output.get("reason", "")).strip():
        fails.append("empty reason")
    return fails


def run_suite(ticket_output_pairs) -> dict:
    total = 0
    failed = 0
    detail = []
    for t, o in ticket_output_pairs:
        f = check(t, o)
        total += 1
        if f:
            failed += 1
            detail.append({"id": t["id"], "fails": f})
    return {"total": total, "passed": total - failed, "failed": failed, "detail": detail}
