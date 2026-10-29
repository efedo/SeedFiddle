"""Spin-box precision derived from a parameter's step and values."""

from __future__ import annotations


def _fraction_digits(value: float, limit: int = 6) -> int:
    text = format(float(value), f".{limit}f").rstrip("0")
    return len(text.split(".", 1)[1]) if "." in text else 0


def spin_decimals(spec, *values) -> int:
    """Decimals implied by a parameter's step, widened to show its values exactly.

    Widening matters: a spin box rounds to its decimals, so a default of 0.125
    shown with two decimals would be written back as 0.13 on the next edit.
    """

    step = float(spec.step or 0.01)
    digits = _fraction_digits(step)
    for value in (*values, spec.minimum, spec.maximum):
        if value is not None:
            digits = max(digits, _fraction_digits(value))
    return min(6, digits)
