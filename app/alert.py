"""Post a Pheme line to Discord when a publish run is not clean.

    python -m app.alert site/regime.json

Silent when the read is OK. Posts when data quality is DEGRADED or STALE, or when the
run produced no regime.json at all. Same conventions as meridian-clock's pheme.py:
discord.com only, Mozilla User-Agent (urllib's default UA gets HTTP 403), never log the URL.
Unset DISCORD_PHEME_WEBHOOK: prints "Pheme silent: no webhook" and exits 0.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

PAGE_URL = "https://chrisrileyy91-gif.github.io/regime-engine/"
MAX_LINES = 8


def build_message(payload: dict[str, Any] | None, run_context: str) -> str | None:
    if payload is None:
        return f"Regime publish FAILED ({run_context}): no read was produced. The page still shows the last good read."
    quality = payload.get("data_quality") or {}
    status = quality.get("status", "OK")
    if status == "OK":
        return None
    lines = [f"Regime read {status} ({run_context}): {quality.get('summary', '')}"]
    warns = [i["message"] for i in quality.get("issues", []) if i.get("severity") == "warn"]
    lines += [f"- {m}" for m in warns[:MAX_LINES]]
    if len(warns) > MAX_LINES:
        lines.append(f"- and {len(warns) - MAX_LINES} more")
    lines.append(PAGE_URL)
    return "\n".join(lines)


def post(webhook: str, content: str) -> None:
    if not webhook.startswith("https://discord.com/"):
        raise ValueError("webhook must be a https://discord.com/ URL")
    body = json.dumps({"username": "Pheme", "content": content[:1900]}).encode()
    request = urllib.request.Request(
        webhook,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 (regime-engine)"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        if response.status not in (200, 204):
            raise RuntimeError(f"Discord returned HTTP {response.status}")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    path = Path(args[0]) if args else Path("site/regime.json")
    run_context = os.environ.get("RUN_CONTEXT", "manual run")
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        payload = None
    message = build_message(payload, run_context)
    if message is None:
        print("Pheme: read is clean, nothing to post")
        return 0
    print(message)
    webhook = os.environ.get("DISCORD_PHEME_WEBHOOK", "")
    if not webhook:
        print("Pheme silent: no webhook")
        return 0
    try:
        post(webhook, message)
    except Exception as exc:
        print(f"Pheme post failed: {type(exc).__name__}", file=sys.stderr)  # never echo the URL
        return 1
    print("Pheme: posted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
