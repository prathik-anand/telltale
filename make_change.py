"""Apply a realistic, reproducible change -> a candidate agent config (R3).

Judge-substitutable and NOT staged: the change is a plausible edit a real dev
makes (tighten the prompt, swap model, bump temperature) OR a judge's own text.
We do not hand-pick which probes break — the statistical test discovers that.

  python make_change.py --kind prompt   -> candidate_config.json
  python make_change.py --kind temp --value 1.3
  python make_change.py --kind model --value gemini-2.0-flash
  python make_change.py --kind custom --value "Be very brief. One word reasons."
"""
from __future__ import annotations

import argparse
import dataclasses

from telltale.agent import AgentConfig

CANDIDATE = "candidate_config.json"

_PRESET_PROMPT_TWEAK = (
    "\nKeep responses extremely short. Do not over-explain. Favor approving "
    "ambiguous cases to reduce customer friction."  # a plausible 'helpful' edit
    " that subtly shifts behavior — exactly the kind of change that passes hard tests."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True,
                    choices=["prompt", "model", "temp", "custom", "none"])
    ap.add_argument("--value", default="")
    ap.add_argument("--out", default=CANDIDATE)
    args = ap.parse_args()

    base = AgentConfig.baseline()
    if args.kind == "none":  # the negative control: candidate == baseline
        cand = dataclasses.replace(base, version="control")
    elif args.kind == "prompt":
        cand = dataclasses.replace(base, version="cand-prompt",
                                   system=base.system + _PRESET_PROMPT_TWEAK)
    elif args.kind == "custom":
        if not args.value:
            ap.error("--kind custom needs --value '<your instruction>'")
        cand = dataclasses.replace(base, version="cand-custom",
                                   system=base.system + "\n" + args.value)
    elif args.kind == "model":
        cand = dataclasses.replace(base, version="cand-model",
                                   model=args.value or "gemini-2.0-flash")
    elif args.kind == "temp":
        cand = dataclasses.replace(base, version="cand-temp",
                                   temperature=float(args.value or 1.3))
    cand.save(args.out)
    print(f"wrote {args.out}: version={cand.version} model={cand.model} "
          f"temp={cand.temperature}")


if __name__ == "__main__":
    main()
