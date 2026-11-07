"""Deterministic fiqh arithmetic - the owner ruling as code.

ARCHITECTURE.md section 11: "Calculation is never delegated to the model."
Inheritance (``mirath``), zakat (nisab / hawl / rate), the camel and cattle
columns and zakat al-fitr go through these two modules as **tools**, exact
``fractions.Fraction`` only. His corpus supplies the rules and conditions;
these functions supply the arithmetic.

Pure: same input, same output, on any machine, forever. No network, no clock,
no model, no floats. Callers may render his wording *around* a result, but
nothing here reads or returns anything a model produced.
"""
from __future__ import annotations

from . import mirath

__all__ = ["mirath"]
