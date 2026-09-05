from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .semantic_geometry import audit_breadth_condition_deltas


def parse_top_actor_mapping(values: Sequence[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        clip_id, separator, actor = value.partition("=")
        if not separator or not clip_id or actor not in {"A1", "A2"}:
            raise ValueError("top-actor values must use clip_id=A1 or clip_id=A2")
        if clip_id in mapping:
            raise ValueError(f"duplicate top-actor clip: {clip_id}")
        mapping[clip_id] = actor
    if len(mapping) < 3:
        raise ValueError("at least three explicit top-actor mappings are required")
    return mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit identity-cancelled semantic geometry across cached clips."
    )
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--top-actor",
        action="append",
        required=True,
        help="Frozen top/control actor mapping in clip_id=A1 form; repeat per clip.",
    )
    parser.add_argument("--context", default="4fps")
    parser.add_argument("--thinking-mode", default="off")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_breadth_condition_deltas(
        Path(args.cache_root),
        Path(args.output),
        top_actor_by_clip=parse_top_actor_mapping(args.top_actor),
        context=args.context,
        thinking_mode=args.thinking_mode,
    )
    print(
        json.dumps(
            {
                "artifact_count": result["artifact_count"],
                "selected_layers": result["selected_layers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
