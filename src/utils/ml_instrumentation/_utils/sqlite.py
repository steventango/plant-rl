from typing import Any


def maybe_quote(v: Any):
    if isinstance(v, str):
        return quote(v)
    return v


def quote(s: str):
    return f"'{s}'"
