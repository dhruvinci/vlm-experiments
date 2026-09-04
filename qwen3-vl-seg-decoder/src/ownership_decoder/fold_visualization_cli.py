from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .decoder_campaign import fold_run_spec_from_dict
from .fold_visualization import render_completed_fold_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one-frame-at-a-time diagnostics for a completed decoder fold."
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--panel-height", type=int, default=320)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.spec.read_text(encoding="utf-8"))
    manifest = render_completed_fold_diagnostics(
        fold_run_spec_from_dict(payload),
        args.output_root,
        panel_size=(args.panel_width, args.panel_height),
    )
    print(json.dumps({"run_name": manifest["run_name"], "frame_count": manifest["frame_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
