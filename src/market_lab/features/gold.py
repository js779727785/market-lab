from __future__ import annotations

from decimal import Decimal


def gold_cny_return_components(
    *,
    xau_usd_start: float,
    xau_usd_end: float,
    usd_cny_start: float,
    usd_cny_end: float,
) -> dict[str, float]:
    if min(xau_usd_start, xau_usd_end, usd_cny_start, usd_cny_end) <= 0:
        raise ValueError("gold and FX prices must be positive")
    gold_return_decimal = Decimal(str(xau_usd_end)) / Decimal(str(xau_usd_start)) - 1
    fx_return_decimal = Decimal(str(usd_cny_end)) / Decimal(str(usd_cny_start)) - 1
    gold_return = float(gold_return_decimal)
    fx_return = float(fx_return_decimal)
    return {
        "gold_usd_return": gold_return,
        "fx_return": fx_return,
        "gold_cny_return": float((1 + gold_return_decimal) * (1 + fx_return_decimal) - 1),
    }
