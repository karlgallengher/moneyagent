from __future__ import annotations

from typing import Any


def _number(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def calculate_finance(operation: str, **kwargs: Any) -> dict[str, Any]:
    operation = str(operation or "").strip().lower()
    if operation == "difference":
        a = _number(kwargs.get("a"), "a")
        b = _number(kwargs.get("b"), "b")
        return {"operation": operation, "result": a - b, "formula": "a - b"}

    if operation == "sum":
        values = kwargs.get("values") or []
        if not isinstance(values, list):
            raise ValueError("values must be a list")
        result = sum(_number(value, "value") for value in values)
        return {"operation": operation, "result": result, "formula": "sum(values)"}

    if operation == "ratio":
        numerator = _number(kwargs.get("numerator"), "numerator")
        denominator = _number(kwargs.get("denominator"), "denominator")
        if denominator == 0:
            raise ValueError("denominator must not be zero")
        return {
            "operation": operation,
            "result": numerator / denominator,
            "percent": numerator / denominator * 100,
            "formula": "numerator / denominator",
        }

    if operation == "yoy_reverse":
        current = _number(kwargs.get("current"), "current")
        growth_rate = _number(kwargs.get("growth_rate"), "growth_rate")
        base = current / (1 + growth_rate)
        return {
            "operation": operation,
            "result": base,
            "formula": "current / (1 + growth_rate)",
        }

    if operation == "bond_interest":
        principal = _number(kwargs.get("principal", kwargs.get("B")), "principal")
        rate = _number(kwargs.get("rate", kwargs.get("i")), "rate")
        days = _number(kwargs.get("days", kwargs.get("t")), "days")
        result = principal * rate * days / 365
        return {
            "operation": operation,
            "result": result,
            "formula": "IA = B * i * t / 365",
        }

    if operation == "deductible_claim":
        expense = _number(kwargs.get("expense"), "expense")
        deductible = _number(kwargs.get("deductible"), "deductible")
        compensation = _number(kwargs.get("compensation", 0), "compensation")
        payout_rate = _number(kwargs.get("payout_rate", 1), "payout_rate")
        base = max(0.0, expense - compensation - deductible)
        return {
            "operation": operation,
            "claim_base": base,
            "result": base * payout_rate,
            "formula": "max(0, expense - compensation - deductible) * payout_rate",
        }

    supported = ["difference", "sum", "ratio", "yoy_reverse", "bond_interest", "deductible_claim"]
    raise ValueError(f"unsupported operation: {operation}. supported={supported}")
