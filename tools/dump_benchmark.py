from __future__ import annotations

import argparse
from pathlib import Path

from eval.benchmark import load_benchmark_csv
from eval.utils import normalize_identifier


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dump_benchmark",
        description="Debug tool: print benchmark identifiers (raw + canonical).",
    )
    p.add_argument(
        "--benchmark_csv",
        required=True,
        help="Path to benchmark CSV.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.benchmark_csv).expanduser().resolve()
    benchmark = load_benchmark_csv(path)
    for b in benchmark:
        raw = str(b.canonical_id).strip()
        canon = normalize_identifier(raw)
        print(f"{raw}\t{canon}")
    print(f"rows={len(benchmark)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


