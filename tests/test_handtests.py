"""Deterministic proof of the demo's gut-punch (no model key needed):
the hand-written suite passes on BOTH a well-formed baseline-shaped output
and a semantically-regressed-but-schema-valid output. That is exactly why
hard assertions can't catch non-deterministic agent drift.
"""
from telltale.handtests import check, run_suite

TICKET = {"id": "RB-1000", "item": "webcam", "requested": 100.0,
          "days": 40, "receipt": True, "condition": "opened-unused",
          "manager_override": False}

# Baseline-shaped: policy says 31-60 days -> store credit; well-reasoned deny of cash
BASELINE_OUT = {"decision": "deny", "amount": 0.0,
                "reason": "Item is 40 days old; policy allows store credit only, not a cash refund."}

# Semantically regressed: agent got laxer/sloppier in REASONING but output is
# still schema-valid and policy-consistent enough to pass hard asserts.
REGRESSED_OUT = {"decision": "deny", "amount": 0.0, "reason": "no"}


def test_handtests_pass_on_baseline():
    assert check(TICKET, BASELINE_OUT) == []


def test_handtests_ALSO_pass_on_semantic_regression():
    # The whole thesis: a degraded reason ("no") still passes every hard assert.
    assert check(TICKET, REGRESSED_OUT) == []


def test_handtests_catch_only_schema_policy_violations():
    bad = {"decision": "approve", "amount": 999.0, "reason": "ok"}  # exceeds requested
    fails = check(TICKET, bad)
    assert "amount exceeds requested" in fails


def test_run_suite_aggregates():
    pairs = [(TICKET, BASELINE_OUT), (TICKET, REGRESSED_OUT)]
    r = run_suite(pairs)
    assert r["total"] == 2 and r["failed"] == 0  # both "pass" — that's the point
