"""Parsing helpers for Fortran-formatted numbers."""

from __future__ import annotations

import re

import numpy as np

# "1.0D+01" -> "1.0E+01"; "1.234-100" (3-digit exponent without 'E') -> "1.234E-100".
_D_EXPONENT = re.compile(r"[dD]")
_BARE_EXPONENT = re.compile(r"(?<=[0-9.])([+-]\d{3})(?=\s|$)")


def fortran_float(token: str) -> float:
    text = _D_EXPONENT.sub("E", token.strip())
    try:
        return float(text)
    except ValueError:
        return float(_BARE_EXPONENT.sub(r"E\1", text))


def parse_floats(text: str, count: int) -> np.ndarray:
    """Parse whitespace-separated Fortran reals; raises if fewer than ``count`` values."""
    tokens = _D_EXPONENT.sub("E", text).split()[:count]
    try:
        array = np.array(tokens, dtype=np.float64)
    except ValueError:
        array = np.array([fortran_float(token) for token in tokens], dtype=np.float64)
    if array.size != count:
        raise ValueError(f"expected {count} values, found {array.size}")
    return array


def parse_ints(text: str, count: int) -> np.ndarray:
    tokens = text.split()[:count]
    if len(tokens) != count:
        raise ValueError(f"expected {count} values, found {len(tokens)}")
    return np.array(tokens, dtype=np.int64)
