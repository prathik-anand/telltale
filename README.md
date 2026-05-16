# Telltale

> Your AI agent passes every test you wrote — and still broke production yesterday.

Telltale is an agentic regression-tester for **non-deterministic** AI agents. Hand-written
assertions can't catch an agent that answers differently every run; Telltale runs a
**pre-registered statistical drift test** over real model behavior and opens a case only
when the evidence — not a tuned threshold — says the agent regressed.

Built for **UiPath AgentHack 2026** (Agentic Testing track). On the platform path the
tester loop runs as a UiPath **Maestro** agentic process and each probe executes as a
real **Test Cloud** run; a regression opens a **Maestro Case** with a human gate.

## Why you can trust the verdict (this is the whole point)

- The regression decision is a **Mann-Whitney U test + Benjamini-Hochberg FDR + an
  effect-size gate**, with `alpha`/`delta` **pre-registered in `prereg.yaml`** and
  git-tracked *before* any demo. It is not a score calibrated to look good.
- A **negative control** runs every time: baseline-vs-baseline must flag **zero**
  probes. A detector that "always finds something" fails its own control and is
  rejected.
- Probes are **frozen and judge-substitutable** (`data/probes.json`) — bring your own.
- **No mock path.** No key → it fails closed with an error, never a fake result.
  (`telltale/config.py: MissingCredential`.) The math is real and unit-tested
  without a key; the model calls are real Gemini 2.5 Pro.

## Run it (60 seconds once the key is set)

```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env            # then put your Gemini API key in .env
python -m telltale.probes       # freeze the 60-probe set
python make_change.py --kind prompt          # produce a candidate (or --kind temp/model/custom)
python -m telltale.run --candidate candidate_config.json
# -> runs_out/verdict.json : regressed probes, p-values, effect sizes, negative-control status
```

`GEMINI_API_KEY` (Google AI Studio) powers both the target agent (`gemini-2.5-pro`)
and the embeddings (`gemini-embedding-001`) — one key.

## What's real vs. pending

| Piece | State |
|---|---|
| Statistical drift engine (`telltale/drift.py`) | **real, unit-tested, no key needed** (`tests/test_drift.py`) |
| Hand-test "lie" demonstration (`telltale/handtests.py`) | **real, unit-tested** (`tests/test_handtests.py`) |
| Target agent + embeddings (real Gemini, fail-closed) | **real, runs the moment `.env` has a key** |
| UiPath Maestro / Test Cloud wrapper (R2) | pending UiPath Labs creds (≈3-day access form) |

## Tech

Python · Gemini 2.5 Pro + Gemini embeddings · numpy/scipy (Mann-Whitney, bootstrap,
Benjamini-Hochberg) · UiPath Maestro + Test Cloud (platform path). Authored with Claude Code.

## Results — real runs (not staged)

Two real runs on Gemini 2.5 Pro, pre-registered `alpha=0.05 delta=0.04` (committed before the runs), 24 frozen probes × K4. Evidence: [`evidence/`](evidence/).

| Change a dev might ship | Hand-written tests | Telltale verdict | Negative control |
|---|---|---|---|
| Prompt tweak ("be concise, lenient") | pass | **NO_REGRESSION** (no false alarm) | passed |
| Model downgrade `gemini-2.5-pro → 2.5-flash` | **5/5 PASS** | **REGRESSION_DETECTED — 10/24 probes** (p_adj≈0.034) | passed |

The hand-written suite passes on the downgraded model; Telltale's pre-registered statistical
test catches the semantic regression it misses. The negative control (baseline-vs-baseline)
flagged zero on every run — the detector does not cry wolf. That pairing — a true negative
and a true positive, with a passing control — is the integrity claim, demonstrable from
`evidence/`, not asserted.
