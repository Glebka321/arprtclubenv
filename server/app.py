# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Clothing Brand Ctr Env Environment.

This module creates an HTTP server that exposes the ClothingBrandCtrEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e
from fastapi.responses import HTMLResponse

try:
    from ..models import ClothingBrandCtrAction, ClothingBrandCtrObservation
except ImportError:  # pragma: no cover - supports direct server.app imports
    from models import ClothingBrandCtrAction, ClothingBrandCtrObservation
from .clothing_brand_ctr_env_environment import ClothingBrandCtrEnvironment


# Create the app with web interface and README integration
app = create_app(
    ClothingBrandCtrEnvironment,
    ClothingBrandCtrAction,
    ClothingBrandCtrObservation,
    env_name="clothing_brand_ctr_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

LANDING_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Email Campaign Simulation</title>
    <style>
      :root {
        --bg: #f6f4ef;
        --card: #ffffff;
        --ink: #222222;
        --muted: #6e675e;
        --accent: #d5632f;
        --accent-soft: #ffe5d8;
      }
      body {
        margin: 0;
        font-family: "Avenir Next", "Segoe UI", sans-serif;
        background: radial-gradient(circle at 15% 10%, #fff7f1 0%, var(--bg) 45%, #efe9df 100%);
        color: var(--ink);
      }
      .wrap {
        max-width: 980px;
        margin: 0 auto;
        padding: 42px 24px 56px;
      }
      .hero {
        background: var(--card);
        border: 1px solid #ece5dc;
        border-radius: 18px;
        padding: 28px;
        box-shadow: 0 16px 40px rgba(36, 26, 15, 0.08);
      }
      h1 {
        margin: 0 0 10px 0;
        font-size: clamp(28px, 4vw, 44px);
        line-height: 1.1;
      }
      .sub {
        margin: 0;
        color: var(--muted);
        font-size: 16px;
      }
      .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 18px 0 0 0;
      }
      .badge {
        background: var(--accent-soft);
        color: #7a3518;
        border: 1px solid #ffd2bc;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 13px;
        font-weight: 600;
      }
      .grid {
        margin-top: 18px;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 12px;
      }
      .card {
        background: var(--card);
        border: 1px solid #ece5dc;
        border-radius: 14px;
        padding: 14px;
      }
      .card h3 {
        margin: 0 0 8px 0;
        font-size: 15px;
      }
      .card p {
        margin: 0;
        color: var(--muted);
        font-size: 14px;
      }
      .links {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
      }
      .links a {
        text-decoration: none;
        color: #fff;
        background: var(--accent);
        border-radius: 10px;
        padding: 10px 14px;
        font-weight: 700;
        font-size: 14px;
      }
      .links a.secondary {
        background: #2f6ea7;
      }
      pre {
        margin: 16px 0 0 0;
        background: #1a1f24;
        color: #d8e6f3;
        border-radius: 12px;
        padding: 14px;
        overflow-x: auto;
        font-size: 12px;
      }
    </style>
  </head>
  <body>
    <main class="wrap">
      <section class="hero">
        <h1>Email Marketing Campaign Simulator</h1>
        <p class="sub">
          Simulates multi-step brand email campaigns and optimizes opens, click-through rate, and purchases.
          Uses Hugging Face personas and DeepSeek for generation + marketer judging.
        </p>
        <div class="badge-row">
          <span class="badge">5-email schedule optimization</span>
          <span class="badge">Day + time simulation</span>
          <span class="badge">DeepSeek marketer judge</span>
          <span class="badge">Persona-based outcomes</span>
        </div>
        <div class="grid">
          <article class="card">
            <h3>Primary metrics</h3>
            <p>Open rate, CTR, click-to-open rate, purchases, and composite campaign score.</p>
          </article>
          <article class="card">
            <h3>Audience model</h3>
            <p>Supports Hugging Face Nemotron personas and synthetic fallback for experimentation.</p>
          </article>
          <article class="card">
            <h3>Judging layer</h3>
            <p>10x marketer judge combines deterministic checks with LLM scoring for subject/body quality.</p>
          </article>
          <article class="card">
            <h3>Optimization output</h3>
            <p>Top schedule ranking, per-step performance, and top purchasing personas.</p>
          </article>
        </div>
        <div class="links">
          <a href="/docs">Open API Docs</a>
          <a class="secondary" href="/health">Health Check</a>
          <a class="secondary" href="/schema">Schema</a>
        </div>
        <pre>python simulate_5_email_campaign.py --persona-source hf --send-days mon,tue,wed,thu,fri --send-hours 8,10,12,15,18</pre>
      </section>
    </main>
  </body>
</html>
"""


@app.get("/", include_in_schema=False)
def landing_page() -> HTMLResponse:
    return HTMLResponse(LANDING_HTML)


@app.get("/web", include_in_schema=False)
def landing_page_web() -> HTMLResponse:
    return HTMLResponse(LANDING_HTML)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m clothing_brand_ctr_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn clothing_brand_ctr_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
