# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Email Campaign Simulation environment.

This server exposes OpenEnv-compatible endpoints and serves a lightweight
campaign analytics dashboard at "/" and "/web".
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from fastapi.responses import HTMLResponse, JSONResponse

try:
    from ..models import ClothingBrandCtrAction, ClothingBrandCtrObservation
except ImportError:  # pragma: no cover - supports direct server.app imports
    from models import ClothingBrandCtrAction, ClothingBrandCtrObservation
from .clothing_brand_ctr_env_environment import ClothingBrandCtrEnvironment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_INDEX_PATH = PROJECT_ROOT / "web" / "index.html"
EVALS_DIR = PROJECT_ROOT / "outputs" / "evals"


def _safe_float(value: object, default: float = 0.0) -> float:
    """Best-effort float parser."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    """Best-effort integer parser."""
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _read_csv_rows(path: Path, limit: int | None = None) -> List[Dict[str, str]]:
    """Read CSV rows if file exists."""
    if not path.exists():
        return []

    rows: List[Dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            rows.append(dict(row))
            if limit is not None and idx + 1 >= limit:
                break
    return rows


def _format_file_meta(path: Path) -> Dict[str, object]:
    """Return file metadata used by dashboard."""
    if not path.exists():
        return {"path": str(path.relative_to(PROJECT_ROOT)), "exists": False}
    stat = path.stat()
    updated = datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat()
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "size_bytes": stat.st_size,
        "updated_at_utc": updated,
    }


def load_campaign_stats() -> Dict[str, object]:
    """Aggregate latest simulation output files for dashboard display."""
    schedule_path = EVALS_DIR / "five_email_schedule_results.csv"
    step_path = EVALS_DIR / "five_email_best_schedule_steps.csv"
    arm_path = EVALS_DIR / "brand_campaign_arm_results.csv"

    schedule_rows = _read_csv_rows(schedule_path, limit=25)
    step_rows = _read_csv_rows(step_path, limit=25)
    arm_rows = _read_csv_rows(arm_path, limit=25)

    top_schedules = [
        {
            "rank": _safe_int(row.get("rank")),
            "schedule": row.get("schedule", ""),
            "open_rate": _safe_float(row.get("open_rate")),
            "ctr": _safe_float(row.get("ctr")),
            "purchase_rate": _safe_float(row.get("purchase_rate")),
            "composite_score": _safe_float(row.get("composite_score")),
            "generation_source": row.get("generation_source", ""),
            "marketer_score": _safe_float(row.get("marketer_score")),
            "opens": _safe_int(row.get("opens")),
            "clicks": _safe_int(row.get("clicks")),
            "purchases": _safe_int(row.get("purchases")),
        }
        for row in schedule_rows[:10]
    ]
    best_schedule = top_schedules[0] if top_schedules else {}

    step_breakdown = [
        {
            "step_idx": _safe_int(row.get("step_idx")),
            "step_name": row.get("step_name", ""),
            "send_day": row.get("send_day", ""),
            "send_hour": _safe_int(row.get("send_hour")),
            "open_rate": _safe_float(row.get("open_rate")),
            "ctr": _safe_float(row.get("ctr")),
            "purchase_rate": _safe_float(row.get("purchase_rate")),
            "subject_line": row.get("subject_line", ""),
        }
        for row in step_rows
    ]

    top_arms = [
        {
            "rank": _safe_int(row.get("rank")),
            "arm_id": row.get("arm_id", ""),
            "variant_name": row.get("variant_name", ""),
            "brand_voice": row.get("brand_voice", ""),
            "send_hour": _safe_int(row.get("send_hour")),
            "open_rate": _safe_float(row.get("open_rate")),
            "ctr": _safe_float(row.get("ctr")),
            "purchase_rate": _safe_float(row.get("purchase_rate")),
            "composite_score": _safe_float(row.get("composite_score")),
            "marketer_score": _safe_float(row.get("marketer_score")),
            "subject_line": row.get("subject_line", ""),
        }
        for row in arm_rows[:10]
    ]
    best_arm = top_arms[0] if top_arms else {}

    return {
        "status": "ok",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "summary": {
            "best_schedule": best_schedule,
            "best_arm": best_arm,
            "top_schedule_count": len(top_schedules),
            "step_count": len(step_breakdown),
            "top_arm_count": len(top_arms),
        },
        "top_schedules": top_schedules,
        "step_breakdown": step_breakdown,
        "top_arms": top_arms,
        "files": {
            "five_email_schedule_results": _format_file_meta(schedule_path),
            "five_email_best_schedule_steps": _format_file_meta(step_path),
            "brand_campaign_arm_results": _format_file_meta(arm_path),
        },
    }


def _load_dashboard_html() -> str:
    """Load dashboard HTML from disk with safe fallback."""
    if WEB_INDEX_PATH.exists():
        return WEB_INDEX_PATH.read_text(encoding="utf-8")

    return (
        "<!doctype html><html><body><h1>Email Campaign Dashboard Missing</h1>"
        "<p>Create web/index.html in the project root.</p></body></html>"
    )


# Create the app with web interface and README integration
app = create_app(
    ClothingBrandCtrEnvironment,
    ClothingBrandCtrAction,
    ClothingBrandCtrObservation,
    env_name="clothing_brand_ctr_env",
    max_concurrent_envs=1,  # increase for more concurrent WebSocket sessions
)


@app.get("/", include_in_schema=False)
def landing_page() -> HTMLResponse:
    """Dashboard landing page."""
    return HTMLResponse(_load_dashboard_html())


@app.get("/web", include_in_schema=False)
def landing_page_web() -> HTMLResponse:
    """Compatibility route for Space base path."""
    return HTMLResponse(_load_dashboard_html())


@app.get("/campaign/stats")
def campaign_stats() -> JSONResponse:
    """Return latest aggregated campaign stats from simulation CSV outputs."""
    return JSONResponse(load_campaign_stats())


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    Args:
        host: Host address to bind to.
        port: Port number to listen on.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
