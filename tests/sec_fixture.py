"""Real SEC filings (fixtures/sec_filings.txt) as a fundamentals.SecCache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

FIXTURE = Path(__file__).parent / "fixtures" / "sec_filings.txt"


def _date(yymmdd: str) -> str:
    return f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:]}"


def sec_cache(fetched_at: str = "2026-10-14T10:00:00+00:00") -> dict[str, Any]:
    companies: dict[str, Any] = {}
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, tag, rows = line.split()
        ticker, metric = key.split(":")
        facts = []
        for row in rows.split(";"):
            start, end, val = row.split(",")
            facts.append({"start": _date(start), "end": _date(end), "val": float(val), "form": "10-Q", "filed": ""})
        companies.setdefault(ticker, {})[metric] = {"tag": tag, "facts": facts, "latest": facts[-1]["end"]}
    return {"fetched_at": fetched_at, "companies": companies}
