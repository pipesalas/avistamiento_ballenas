"""Measure dataset startup in a fresh process, sampling RSS every 5 ms.

    uv run --with-requirements requirements-dev.txt python scripts/profile_api.py
"""

import json
from pathlib import Path
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    process = psutil.Process()
    peak = process.memory_info().rss
    done = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not done.wait(0.005):
            peak = max(peak, process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    started = time.perf_counter()
    try:
        from api.store import get_store

        store = get_store()
        elapsed = time.perf_counter() - started
        rss = process.memory_info().rss
    finally:
        done.set()
        sampler.join()

    print(json.dumps({
        "startup_seconds": round(elapsed, 3),
        "rss_mib": round(rss / 2**20, 2),
        "sampled_peak_rss_mib": round(max(peak, rss) / 2**20, 2),
        "temperature_rows": len(store.temperature),
        "chlorophyll_rows": len(store.chlorophyll),
    }, indent=2))


if __name__ == "__main__":
    main()
