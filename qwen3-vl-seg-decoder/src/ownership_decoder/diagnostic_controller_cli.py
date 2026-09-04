from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .diagnostic_controller import DiagnosticCampaignSpec, run_diagnostic_campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render all selected decoder folds using serial guarded workers."
    )
    parser.add_argument("--decoder-output-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--child-memory-gib", type=int, default=4)
    parser.add_argument("--min-host-available-gib", type=int, default=2)
    parser.add_argument("--attempts-per-fold", type=int, choices=(1, 2), default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_diagnostic_campaign(
        DiagnosticCampaignSpec(
            decoder_output_root=args.decoder_output_root,
            output_root=args.output_root,
            python_executable=args.python,
            child_memory_max_bytes=args.child_memory_gib * 1024**3,
            min_host_available_bytes=args.min_host_available_gib * 1024**3,
            attempts_per_fold=args.attempts_per_fold,
        )
    )
    print(json.dumps({"status": "complete", "diagnostic_count": result["diagnostic_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
