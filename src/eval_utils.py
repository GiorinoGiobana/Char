from __future__ import annotations

from typing import Any, Union


def _result_is_error(result: Union[list[Any], str]) -> bool:
    return isinstance(result, str) and result.startswith("Error:")


def _cell_equals(a: Any, b: Any, eps: float = 1e-6) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= eps
    return str(a) == str(b)


def _normalize_rows(result: Union[list[Any], str]) -> list[tuple[Any, ...]]:
    if isinstance(result, str):
        return [(result,)]

    rows: list[tuple[Any, ...]] = []
    for r in result:
        if isinstance(r, tuple):
            rows.append(r)
        elif isinstance(r, list):
            rows.append(tuple(r))
        else:
            rows.append((r,))
    return rows


def compare_query_results(
    sql: str,
    gold: Union[list[Any], str],
    candidate: Union[list[Any], str],
) -> tuple[bool, str]:
    if _result_is_error(gold):
        return False, "INVALID_GOLD_SQL"
    if _result_is_error(candidate):
        return False, f"CANDIDATE_ERROR: {candidate}"

    gold_rows = _normalize_rows(gold)
    cand_rows = _normalize_rows(candidate)

    order_matters = "ORDER BY" in sql.upper()
    if not order_matters:
        gold_rows = sorted(gold_rows, key=lambda x: repr(x))
        cand_rows = sorted(cand_rows, key=lambda x: repr(x))

    if len(gold_rows) != len(cand_rows):
        return False, f"ROW_COUNT_MISMATCH: gold={len(gold_rows)} candidate={len(cand_rows)}"

    for i, (gr, cr) in enumerate(zip(gold_rows, cand_rows)):
        if len(gr) != len(cr):
            return False, f"COLUMN_COUNT_MISMATCH_AT_ROW_{i}: gold={len(gr)} candidate={len(cr)}"
        for j, (a, b) in enumerate(zip(gr, cr)):
            if not _cell_equals(a, b):
                return False, f"VALUE_MISMATCH_AT_ROW_{i}_COL_{j}: gold={a} candidate={b}"

    return True, "MATCH"


def truncate_for_json(value: Any, max_chars: int = 5000) -> Any:
    s = str(value)
    if len(s) <= max_chars:
        return value
    return s[:max_chars] + "... [TRUNCATED]"

