"""Discovery endpoints. Both run the real pipeline with the caller's own
session credentials; /stream emits NDJSON progress events."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..deps import (Services, current_session, resolve_credentials, rate_key, require_client_header, services)
from ..errors import RippleError, SerpApiInvalidKey
from ..models import DiscoverRequest, DiscoveryResponse
from ..security import Secret

router = APIRouter(prefix="/api", tags=["discovery"])
logger = logging.getLogger("ripple.discovery")


def _prepare(request: Request, svc: Services):
    sid, data = current_session(request)
    svc.discovery_limiter.check(rate_key(sid))
    import os
    creds = resolve_credentials(svc, data)
    if not creds.serpapi_key:
        env_key = os.getenv("SERPAPI_API_KEY")
        if env_key:
            creds.serpapi_key = Secret(env_key)
    return sid, creds


@router.post("/discover", response_model=DiscoveryResponse,
             dependencies=[Depends(require_client_header)])
async def discover(body: DiscoverRequest, request: Request,
                   svc: Services = Depends(services)) -> DiscoveryResponse:
    sid, creds = _prepare(request, svc)
    try:
        return await svc.engine.run(body.event, creds)
    except SerpApiInvalidKey:
        svc.store.revoke(sid)  # the stored key no longer works; force reconnect
        raise


@router.post("/discover/stream", dependencies=[Depends(require_client_header)])
async def discover_stream(body: DiscoverRequest, request: Request,
                          svc: Services = Depends(services)) -> StreamingResponse:
    print("🚨 WIVVI SEARCH STARTED")
    sid, creds = _prepare(request, svc) # auth errors are returned before streaming
    queue: asyncio.Queue = asyncio.Queue()

    async def progress(stage: str, message: str, detail: dict | None = None) -> None:
        await queue.put({"type": "progress", "stage": stage, "message": message, **(detail or {})})

    async def worker() -> None:
        try:
            print("🕸️ ASKING THE ENGINE (SERPAPI & GEMINI)...")
            result = await svc.engine.run(body.event, creds, progress)
            await queue.put({"type": "result", "data": result.model_dump()})
        except RippleError as exc:
            if isinstance(exc, SerpApiInvalidKey):
                svc.store.revoke(sid)
            await queue.put({"type": "error", **exc.to_dict()})
        except Exception:
            logger.exception("Discovery failed")
            await queue.put({"type": "error", **RippleError().to_dict()})
        finally:
            await queue.put(None)

    async def events():
        task = asyncio.create_task(worker())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield json.dumps(item) + "\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
