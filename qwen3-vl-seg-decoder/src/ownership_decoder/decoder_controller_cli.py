from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .decoder_controller import LocalDecoderCampaignSpec, run_local_decoder_campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the serial, resource-guarded nested decoder campaign."
    )
    parser.add_argument("--reviewed-label-campaign", required=True)
    parser.add_argument("--qwen-breadth-root", required=True)
    parser.add_argument("--qwen-download-manifest", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--child-memory-gib", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_local_decoder_campaign(
        LocalDecoderCampaignSpec(
            reviewed_label_campaign=Path(args.reviewed_label_campaign),
            qwen_breadth_root=Path(args.qwen_breadth_root),
            qwen_download_manifest=Path(args.qwen_download_manifest),
            input_root=Path(args.input_root),
            output_root=Path(args.output),
            python_executable=Path(args.python),
            width=args.width,
            residual_blocks=args.residual_blocks,
            max_epochs=args.max_epochs,
            patience=args.patience,
            gradient_accumulation=args.gradient_accumulation,
            device=args.device,
            use_amp=not args.no_amp,
            child_memory_max_bytes=args.child_memory_gib * 1024**3,
        )
    )
    print(
        json.dumps(
            {
                "scientific_status": result["scientific_status"],
                "north_star": result["north_star"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
