from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .decoder_campaign import fold_run_spec_from_dict, run_decoder_fold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated ownership-decoder fold.")
    parser.add_argument("--spec", required=True, help="Immutable decoder fold work-item JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.spec)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"decoder fold work item is invalid: {path}") from error
    result = run_decoder_fold(fold_run_spec_from_dict(payload))
    print(
        json.dumps(
            {
                "run_name": result["run_name"],
                "heldout_clip": result["heldout_clip"],
                "test_metrics": result["test_metrics"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
