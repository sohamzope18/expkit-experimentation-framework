"""Assert that every study ran and every study passed.

`make sim` runs five studies in sequence. If one crashes, make aborts and the
artifacts from the studies after it are left over from a previous run -- stale
files that look exactly like fresh ones. Grepping the output for "PASS" does not
catch this, because the missing lines are simply absent rather than wrong.

This module checks the artifacts themselves: every expected file exists, and every
study's own pass flag is true. Run as the last step of `make sim`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

# study name -> the boolean keys in its JSON that must all be true
EXPECTED = {
    "study_type1": ["all_within_3_mc_se"],
    "study_coverage": ["delta_method_holds_nominal"],
    "study_power": ["all_within_3_mc_se"],
    "study_cuped": ["reduction_matches_theory", "estimator_unbiased", "guard_calibrated"],
    "study_peeking": ["sequential_controls_type1", "peeking_inflates_type1"],
}


def main() -> int:
    problems = []
    for study, flags in EXPECTED.items():
        payload_path = RESULTS / f"{study}.json"
        chart_path = RESULTS / f"{study}.png"
        if not payload_path.exists():
            problems.append(f"{study}: {payload_path.name} is missing -- the study did not run")
            continue
        if not chart_path.exists():
            problems.append(f"{study}: {chart_path.name} is missing")
        payload = json.loads(payload_path.read_text())
        for flag in flags:
            if flag not in payload:
                problems.append(f"{study}: JSON has no '{flag}' key")
            elif not payload[flag]:
                problems.append(f"{study}: '{flag}' is false")

    if problems:
        print("\nRESULTS VERIFICATION FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"\nAll {len(EXPECTED)} studies produced artifacts and reported passing checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
