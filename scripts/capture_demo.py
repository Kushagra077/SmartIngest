"""Regenerate the demo screenshots in ``demo/`` automatically.

Starts the API + Streamlit UI in offline mock mode, drives the UI with a
headless browser, and saves a screenshot of each routing outcome. Re-runnable
and self-contained — it manages its own servers and temp data.

Usage:
    uv sync --group capture
    uv run python -m playwright install chromium
    uv run python scripts/capture_demo.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demo"
VIEWPORT = {"width": 1100, "height": 900}

# (button label substring, banner text to wait for, output filename)
SHOTS = [
    ("Clean invoice", "Auto-approved", "01-auto-approve.png"),
    ("Unknown vendor", "Flagged for review", "02-flag-review.png"),
    ("Injection attempt", "Rejected", "03-reject.png"),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, timeout: float = 40.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(url, timeout=2).status == 200:
                return
        except Exception:  # noqa: BLE001 - server still booting
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}")


def _start_servers(tmp: Path) -> tuple[subprocess.Popen, subprocess.Popen, int]:
    api_port, ui_port = _free_port(), _free_port()
    env = {
        **os.environ,
        "SMARTINGEST_MOCK_LLM": "true",
        "GEMINI_API_KEY": "",
        "SMARTINGEST_DB_PATH": str(tmp / "jobs.db"),
        "SMARTINGEST_UPLOAD_DIR": str(tmp / "uploads"),
        "SMARTINGEST_RATE_LIMIT_PER_MINUTE": "1000",
    }
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "smartingest.api:app",
         "--app-dir", "src", "--port", str(api_port), "--log-level", "warning"],
        cwd=ROOT, env=env,
    )
    _wait_http(f"http://localhost:{api_port}/healthz")

    ui_env = {**env, "SMARTINGEST_API_URL": f"http://localhost:{api_port}"}
    ui = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/frontend/streamlit_app.py",
         "--server.port", str(ui_port), "--server.headless", "true",
         "--server.address", "127.0.0.1"],
        cwd=ROOT, env=ui_env,
    )
    _wait_http(f"http://localhost:{ui_port}/")
    return api, ui, ui_port


def _capture(ui_port: int) -> None:
    from playwright.sync_api import sync_playwright

    DEMO_DIR.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT)
        page.goto(f"http://localhost:{ui_port}/", wait_until="networkidle")
        page.get_by_text("Backend online").wait_for(timeout=20_000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(DEMO_DIR / "00-landing.png"))
        print("  saved 00-landing.png")

        for label, banner, filename in SHOTS:
            page.evaluate("window.scrollTo(0, 0)")
            page.locator("button", has_text=label).first.click()
            page.get_by_text(banner, exact=False).wait_for(timeout=30_000)
            page.get_by_text("Extracted fields").wait_for(timeout=10_000)
            page.wait_for_timeout(600)
            page.evaluate("window.scrollTo(0, 0)")
            page.screenshot(path=str(DEMO_DIR / filename))
            print(f"  saved {filename}")

        browser.close()


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        api, ui, ui_port = _start_servers(Path(td))
        try:
            print("Capturing demo screenshots...")
            _capture(ui_port)
        finally:
            for proc in (ui, api):
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
    print(f"Done. Screenshots written to {DEMO_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
