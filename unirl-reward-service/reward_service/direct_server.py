"""Single-scorer HTTP server without a nested Ray runtime.

Used by UniRL's rank-affine managed reward backend: every train Worker starts
one process in the reward-specific Python environment, inherited onto that
Worker's CUDA-visible device. The process owns one scorer for its full lifetime.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import ctypes
import io
import json
import os
import signal
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from PIL import Image

from reward_service.logging_utils import get_logger
from reward_service.schemas import RewardRequest, ScoreRequest, ScoreResponse
from reward_service.scorers import ScoreItem
from reward_service.scorers.registry import SCORER_MODULES, _try_import, get_scorer_cls

logger = get_logger(__name__)


def _arm_parent_death_signal() -> None:
    parent_pid = int(os.environ.get("UNIRL_REWARD_PARENT_PID", "0") or 0)
    if parent_pid <= 0:
        return
    libc = ctypes.CDLL(None)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        raise OSError("prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent_pid:
        raise SystemExit("reward parent exited before child initialization")


def _request_to_item(request: RewardRequest) -> ScoreItem:
    if not request.history:
        raise HTTPException(status_code=400, detail="history must not be empty")
    history: list[tuple[str, Image.Image | None]] = []
    for turn in request.history:
        image = None
        if turn.image_b64 is not None:
            try:
                image = Image.open(io.BytesIO(base64.b64decode(turn.image_b64))).convert("RGB")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid image_b64: {exc}") from exc
        if turn.video_b64 is not None or turn.video_path is not None:
            raise HTTPException(status_code=400, detail="direct image scorer does not accept video inputs")
        history.append((turn.text, image))
    return ScoreItem(history=history, metadata=request.metadata)


def create_direct_app(scorer_name: str, params: dict[str, Any]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        module_path = SCORER_MODULES.get(scorer_name)
        if module_path:
            _try_import(module_path)
        scorer_cls = get_scorer_cls(scorer_name)
        logger.info("direct scorer loading name=%s params=%s", scorer_name, params)
        app.state.scorer = await asyncio.to_thread(scorer_cls, **params)
        logger.info("direct scorer ready name=%s", scorer_name)
        try:
            yield
        finally:
            scorer = app.state.scorer
            close = getattr(scorer, "close", None)
            if callable(close):
                await asyncio.to_thread(close)

    app = FastAPI(title=f"Direct Reward Service ({scorer_name})", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "rewards": {scorer_name: ["ready"]}}

    @app.get("/rewards")
    async def rewards() -> dict:
        return {"rewards": [scorer_name]}

    @app.post("/score", response_model=ScoreResponse)
    async def score(body: ScoreRequest) -> ScoreResponse:
        if not body.requests:
            return ScoreResponse(results=[], errors=[])
        for request in body.requests:
            unknown = [name for name in request.required_rewards if name != scorer_name]
            if unknown:
                raise HTTPException(status_code=400, detail=f"unknown rewards for this worker: {unknown}")

        items = await asyncio.to_thread(lambda: [_request_to_item(request) for request in body.requests])
        try:
            scores = await asyncio.to_thread(app.state.scorer.score, items)
        except Exception as exc:
            logger.exception("direct scorer failed: %s", exc)
            error = repr(exc)
            return ScoreResponse(
                results=[{} for _ in body.requests],
                errors=[{scorer_name: error} for _ in body.requests],
            )
        if len(scores) != len(body.requests):
            raise RuntimeError(
                f"direct scorer returned {len(scores)} results for {len(body.requests)} requests"
            )
        return ScoreResponse(
            results=[{scorer_name: dict(result)} for result in scores],
            errors=[{} for _ in scores],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--params-json", required=True)
    parser.add_argument("--fd", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    _arm_parent_death_signal()
    params = json.loads(args.params_json)
    if not isinstance(params, dict):
        raise TypeError("--params-json must decode to an object")
    app = create_direct_app(args.scorer, params)

    import uvicorn

    if args.fd is not None:
        uvicorn.run(app, fd=args.fd, log_level=args.log_level)
    else:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
