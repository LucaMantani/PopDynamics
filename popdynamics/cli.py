"""Command-line entry point for ``pop-dyn``."""

import sys


def main(argv: list[str] | None = None) -> int:
    print(
        "pop-dyn: the runcard interface is not implemented yet.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
