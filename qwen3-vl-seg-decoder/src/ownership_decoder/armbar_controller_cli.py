from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .armbar_controller import ArmbarCampaignSpec, run_armbar_exploratory_campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the serial, resource-guarded legacy-armbar decoder campaign."
    )
    parser.add_argument("--label-manifest", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--frame-manifest", required=True)
    parser.add_argument("--frame-project-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.60)
    parser.add_argument("--child-memory-gib", type=float, default=4.0)
    parser.add_argument("--min-host-available-gib", type=float, default=4.0)
    parser.add_argument("--min-swap-free-gib", type=float, default=3.0)
    parser.add_argument("--min-gpu-free-mib", type=float, default=1536.0)
    parser.add_argument("--max-gpu-used-fraction", type=float, default=0.75)
    parser.add_argument("--max-job-minutes", type=float, default=30.0)
    parser.add_argument("--resource-poll-seconds", type=float, default=1.0)
    parser.add_argument("--terminate-grace-seconds", type=float, default=10.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_armbar_exploratory_campaign(
        ArmbarCampaignSpec(
            label_manifest=Path(args.label_manifest),
            cache_root=Path(args.cache_root),
            frame_manifest=Path(args.frame_manifest),
            frame_project_root=Path(args.frame_project_root),
            output_root=Path(args.output),
            python_executable=Path(args.python),
            width=args.width,
            residual_blocks=args.residual_blocks,
            max_epochs=args.max_epochs,
            patience=args.patience,
            gradient_accumulation=args.gradient_accumulation,
            device=args.device,
            use_amp=not args.no_amp,
            cuda_memory_fraction=args.cuda_memory_fraction,
            child_memory_max_bytes=int(args.child_memory_gib * 1024**3),
            min_host_available_bytes=int(args.min_host_available_gib * 1024**3),
            min_swap_free_bytes=int(args.min_swap_free_gib * 1024**3),
            min_gpu_free_bytes=int(args.min_gpu_free_mib * 1024**2),
            max_gpu_used_fraction=args.max_gpu_used_fraction,
            maximum_job_runtime_seconds=args.max_job_minutes * 60.0,
            resource_poll_interval_seconds=args.resource_poll_seconds,
            terminate_grace_seconds=args.terminate_grace_seconds,
        )
    )
    print(
        json.dumps(
            {
                "supervision_status": result["supervision_status"],
                "north_star_eligible": result["north_star_eligible"],
                "selected_static_arm": result["selected_static_arm"],
                "selected_action_language_layer": result[
                    "selected_action_language_layer"
                ],
                "exploratory_signal": result["exploratory_signal"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
