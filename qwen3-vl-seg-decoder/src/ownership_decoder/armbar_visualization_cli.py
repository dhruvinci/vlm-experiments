from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .armbar_exploratory import armbar_job_spec_from_dict
from .armbar_visualization import render_armbar_contact_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render contact diagnostics from one completed armbar decoder job."
    )
    parser.add_argument("--spec", required=True, help="Immutable armbar work-item JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--substitution")
    parser.add_argument("--panel-width", type=int, default=360)
    parser.add_argument("--panel-height", type=int, default=640)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec_path = Path(args.spec)
    try:
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"armbar diagnostic work item is invalid: {spec_path}") from error
    result = render_armbar_contact_diagnostics(
        armbar_job_spec_from_dict(payload),
        args.output,
        substitution=args.substitution,
        panel_size=(args.panel_width, args.panel_height),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
