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
import importlib
import io
import json
import math
import os
import signal
import socket
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from PIL import Image

from reward_service.logging_utils import get_logger
from reward_service.schemas import RewardRequest, ScoreRequest, ScoreResponse
from reward_service.scorers import ScoreItem
from reward_service.scorers.registry import SCORER_MODULES, get_scorer_cls

logger = get_logger(__name__)

_PR_SET_PDEATHSIG = 1


def _arm_parent_death_signal() -> None:
    parent_pid = int(os.environ.get("UNIRL_REWARD_PARENT_PID", "0") or 0)
    if parent_pid <= 0:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
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


def _normalize_score(result: Any) -> tuple[dict[str, float], str | None]:
    """Convert scorer output to finite JSON floats or an item-level error."""
    if not isinstance(result, Mapping):
        return {}, f"scorer returned {type(result).__name__}, expected a metric mapping"
    normalized: dict[str, float] = {}
    invalid: list[str] = []
    for name, value in result.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid.append(str(name))
            continue
        if not math.isfinite(numeric):
            invalid.append(str(name))
            continue
        normalized[str(name)] = numeric
    if invalid:
        return {}, f"non-finite or non-numeric metrics: {sorted(invalid)}"
    return normalized, None


def create_direct_app(scorer_name: str, params: dict[str, Any]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        module_path = SCORER_MODULES.get(scorer_name)
        if module_path:
            importlib.import_module(module_path)
        scorer_cls = get_scorer_cls(scorer_name)
        logger.info("direct scorer loading name=%s params=%s", scorer_name, params)
        app.state.scorer = await asyncio.to_thread(scorer_cls, **params)
        app.state.score_lock = asyncio.Lock()
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
        async with app.state.score_lock:
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
            raise RuntimeError(f"direct scorer returned {len(scores)} results for {len(body.requests)} requests")
        results: list[dict[str, dict[str, float]]] = []
        errors: list[dict[str, str]] = []
        for result in scores:
            normalized, error = _normalize_score(result)
            results.append({scorer_name: normalized} if error is None else {})
            errors.append({} if error is None else {scorer_name: error})
        return ScoreResponse(results=results, errors=errors)

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

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level)
    server = uvicorn.Server(config)
    if args.fd is None:
        server.run()
        return
    with socket.socket(fileno=args.fd) as listener:
        server.run(sockets=[listener])


if __name__ == "__main__":
    main()
