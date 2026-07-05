"""Interactive prompt helpers with injectable input for testability."""

from __future__ import annotations

from collections.abc import Callable, Sequence


def prompt_int(
    label: str,
    default: int,
    min_value: int = 1,
    input_fn: Callable[[str], str] = input,
) -> int:
    """Prompt for an integer, showing the default; re-asks until valid.

    Args:
        label: Prompt label.
        default: Value returned on empty input.
        min_value: Minimum accepted value (inclusive).
        input_fn: Injectable input function (tests pass a fake).

    Returns:
        The validated integer.
    """
    while True:
        raw = input_fn(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print(f"  expected an integer >= {min_value}, got {raw!r}")
            continue
        if value < min_value:
            print(f"  expected an integer >= {min_value}, got {value}")
            continue
        return value


def prompt_float(
    label: str,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
    input_fn: Callable[[str], str] = input,
) -> float:
    """Prompt for a float within optional bounds; re-asks until valid."""
    bounds = ""
    if min_value is not None or max_value is not None:
        bounds = f" ({min_value if min_value is not None else '-inf'}"
        bounds += f" .. {max_value if max_value is not None else '+inf'})"
    while True:
        raw = input_fn(f"{label}{bounds} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print(f"  expected a number, got {raw!r}")
            continue
        if min_value is not None and value < min_value:
            print(f"  value must be >= {min_value}")
            continue
        if max_value is not None and value > max_value:
            print(f"  value must be <= {max_value}")
            continue
        return value


def prompt_choice(
    label: str,
    choices: Sequence[str],
    default: str,
    input_fn: Callable[[str], str] = input,
) -> str:
    """Prompt for one of ``choices`` (prefix matching allowed); re-asks until valid."""
    choices_text = ", ".join(choices)
    while True:
        raw = input_fn(f"{label} ({choices_text}) [{default}]: ").strip()
        if not raw:
            return default
        if raw in choices:
            return raw
        matches = [c for c in choices if c.startswith(raw)]
        if len(matches) == 1:
            return matches[0]
        print(f"  expected one of: {choices_text}")


def prompt_confirm(
    label: str,
    default: bool = True,
    input_fn: Callable[[str], str] = input,
) -> bool:
    """Yes/no confirmation; empty input returns the default."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        raw = input_fn(f"{label} {suffix}: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes", "o", "oui"):
            return True
        if raw in ("n", "no", "non"):
            return False
        print("  expected y or n")
