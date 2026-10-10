"""Bounded Yahoo retries. No alternate instruments, identities, or providers."""
import json
import math
import time
from pathlib import Path

import pandas as pd

from .common import json_write

RATE_DELAYS = (300, 600)
ERROR_DELAYS = (15, 45)
MARKET_DELAY = 5


class DownloadRateLimited(RuntimeError):
    pass


def is_rate_limit(error):
    message = str(error).lower()
    return (type(error).__name__ == "YFRateLimitError" or "too many requests" in message
            or "rate limited" in message or "rate limit" in message or "http 429" in message)


def wait_seconds(seconds):
    """Local runner countdown; each individual sleep is at most 30 seconds."""
    remaining = max(0, int(math.ceil(seconds)))
    while remaining:
        print(f"Waiting {remaining}s before the next request. Keep this window open.", flush=True)
        step = min(30, remaining)
        time.sleep(step)
        remaining -= step


def wait_cooldown(path):
    path = Path(path)
    if path.exists():
        metadata = json.loads(path.read_text(encoding="utf-8"))
        remaining = metadata["not_before_unix"] - time.time()
        if remaining > 0:
            print("A saved Yahoo cooldown is still active; restarting does not skip it.", flush=True)
            wait_seconds(remaining)


def save_cooldown(path, delay, market, error):
    deadline = time.time() + delay
    json_write(path, {"not_before_unix": deadline,
                     "not_before_utc": pd.Timestamp(deadline, unit="s", tz="UTC").isoformat(),
                     "market": market["id"], "last_error": str(error),
                     "scope": "conservative local retry policy; not a provider-guaranteed unblock time"})


def fetch_history(market, cfg, raw_dir):
    import yfinance as yf
    cooldown = Path(raw_dir) / "yahoo_cooldown.json"
    wait_cooldown(cooldown)
    for attempt in range(3):
        print(f"Request {market['id']} ({market['ticker']}), attempt {attempt+1}/3", flush=True)
        try:
            # Ticker.history propagates rate-limit errors; download() may log and return an empty frame.
            data = yf.Ticker(market["ticker"]).history(
                start=cfg["download_start"], end=cfg["download_end_exclusive"],
                interval="1d", auto_adjust=True, repair=False, actions=False,
                timeout=30, raise_errors=True)
            if data is None or len(data) < 100:
                raise ValueError("Provider returned empty/insufficient history (not an inferred rate-limit diagnosis).")
            cooldown.unlink(missing_ok=True)
            return data
        except Exception as error:
            limited = is_rate_limit(error)
            if limited:
                delay = RATE_DELAYS[min(attempt, len(RATE_DELAYS)-1)]
                save_cooldown(cooldown, delay, market, error)
            if attempt == 2:
                if limited:
                    raise DownloadRateLimited(
                        f"Yahoo is still rate limiting {market['id']} ({market['ticker']}). "
                        "All further Yahoo requests are stopped. Completed snapshots are preserved. "
                        "A later retry will honor the saved cooldown; alternatively import the same instrument's adjusted CSV. "
                        f"Original error: {type(error).__name__}: {error}") from error
                raise RuntimeError(f"Download failed for {market['id']} ({market['ticker']}): "
                                   f"{type(error).__name__}: {error}") from error
            delay = RATE_DELAYS[attempt] if limited else ERROR_DELAYS[attempt]
            print(f"{'YAHOO RATE LIMIT' if limited else 'DOWNLOAD ERROR'}: {type(error).__name__}: {error}", flush=True)
            wait_seconds(delay)
