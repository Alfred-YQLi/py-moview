from __future__ import annotations

def _as_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "E"))
