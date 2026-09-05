from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .armbar_exploratory import armbar_job_spec_from_dict, run_armbar_job


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one isolated legacy-armbar exploratory decoder job."
    )
    parser.add_argument("--spec", required=True, help="Immutable armbar work-item JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.spec)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"armbar work item is invalid: {path}") from error
    result = run_armbar_job(armbar_job_spec_from_dict(payload))
    print(
        json.dumps(
            {
                "run_name": result["run_name"],
                "evaluation_subset": result["evaluation_subset"],
                "evaluation_metrics": result["evaluation_metrics"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
