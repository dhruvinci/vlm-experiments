from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .breadth_labels import (
    collect_breadth_candidate_inputs,
    finalize_review_manifest,
    freeze_reviewed_label_package,
    verify_reviewed_label_manifest,
    write_candidate_review_package,
    write_review_template,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and freeze the human-reviewed breadth ownership labels."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    candidates = commands.add_parser("candidates")
    candidates.add_argument("--config", action="append", required=True)
    candidates.add_argument("--input-root", required=True)
    candidates.add_argument("--mask-campaign-root", required=True)
    candidates.add_argument("--qwen-breadth-root", required=True)
    candidates.add_argument("--output", required=True)
    candidates.add_argument("--spatial-layer", type=int, default=11)
    candidates.add_argument("--preview-width", type=int, default=480)
    candidates.add_argument("--dilation-radius", type=int, default=31)

    template = commands.add_parser("review-template")
    template.add_argument("--candidate-manifest", required=True)
    template.add_argument("--output", required=True)

    finalize = commands.add_parser("finalize-review")
    finalize.add_argument("--candidate-manifest", required=True)
    finalize.add_argument("--review-manifest", required=True)
    finalize.add_argument("--reviewer", required=True)
    finalize.add_argument(
        "--attest",
        action="store_true",
        help="Attest that every actor identity and contact region was visually reviewed.",
    )

    freeze = commands.add_parser("freeze")
    freeze.add_argument("--candidate-manifest", required=True)
    freeze.add_argument("--review-manifest", required=True)
    freeze.add_argument("--output", required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--label-manifest", required=True)
    verify.add_argument("--clip-id")
    verify.add_argument("--frame-count", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "candidates":
        inputs = collect_breadth_candidate_inputs(
            [Path(value) for value in args.config],
            input_root=Path(args.input_root),
            mask_campaign_root=Path(args.mask_campaign_root),
            qwen_breadth_root=Path(args.qwen_breadth_root),
            spatial_layer=args.spatial_layer,
        )
        result = write_candidate_review_package(
            inputs,
            Path(args.output),
            preview_width=args.preview_width,
            dilation_radius=args.dilation_radius,
        )
    elif args.command == "review-template":
        result = write_review_template(
            Path(args.candidate_manifest),
            Path(args.output),
        )
    elif args.command == "finalize-review":
        result = finalize_review_manifest(
            Path(args.candidate_manifest),
            Path(args.review_manifest),
            reviewer=args.reviewer,
            attested=args.attest,
        )
    elif args.command == "freeze":
        result = freeze_reviewed_label_package(
            Path(args.candidate_manifest),
            Path(args.review_manifest),
            Path(args.output),
        )
    else:
        result = verify_reviewed_label_manifest(
            Path(args.label_manifest),
            expected_clip_id=args.clip_id,
            expected_frame_count=args.frame_count,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
