"""taxi-driver entry point: parse arguments and dispatch subcommands."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from src.cli.commands import cmd_benchmark, cmd_compare, cmd_eval, cmd_play, cmd_train
from src.cli.parser import build_parser

_DISPATCH = {
    "train": cmd_train,
    "eval": cmd_eval,
    "play": cmd_play,
    "benchmark": cmd_benchmark,
    "compare": cmd_compare,
}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (console script ``taxi-driver``)."""
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.command](args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
