"""HTTP surface.

Deliberately thin: it tokenizes nothing, schedules nothing, and holds no state.
Its only jobs are to translate JSON into a `Request`, hand it to whichever
engine is configured, and stream the result back. Swapping engines changes no
line in this file, which is what makes the milestone-to-milestone load-test
comparison an apples-to-apples one.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi import Request as HTTPRequest
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .backends.base import SamplingParams
from .config import Config
from .engine.base import ContextOverflow, QueueFull
from .engine.request import Request
from .metrics import MetricsRegistry


class Message(BaseModel):
    role: str = "user"
    content: str = ""


class GenerateRequest(BaseModel):
    prompt: str | None = None
    messages: list[Message] | None = None
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0)
    repeat_penalty: float = Field(default=1.1, ge=0.0, le=2.0)
    seed: int = 0
    stop: list[str] = Field(default_factory=list)
    ignore_eos: bool = Field(default=False, description="generate exactly max_tokens; for benchmarking")
    stream: bool = False
    priority: int = Field(default=0, description="lower value = higher priority")

    def sampling(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repeat_penalty=self.repeat_penalty,
            seed=self.seed,
            max_tokens=self.max_tokens,
            stop=tuple(self.stop),
            ignore_eos=self.ignore_eos,
        ).normalized()


def build_backend(cfg: Config):
    if cfg.backend == "mock":
        from .backends.mock import MockBackend

        return MockBackend(
            n_ctx=cfg.n_ctx_per_seq * (cfg.max_seqs + cfg.cache_seqs),
            max_seqs=cfg.max_seqs,
            cache_seqs=cfg.cache_seqs if cfg.enable_prefix_cache else 0,
            block_size=cfg.block_size,
            token_latency_s=cfg.mock_token_latency_s,
        )
    from .backends.llama_cpp_backend import LlamaCppBackend

    return LlamaCppBackend(
        model_path=cfg.model_path,
        n_ctx_per_seq=cfg.n_ctx_per_seq,
        max_seqs=cfg.max_seqs,
        cache_seqs=cfg.cache_seqs if cfg.enable_prefix_cache else 0,
        n_gpu_layers=cfg.n_gpu_layers,
        block_size=cfg.block_size,
        verbose=cfg.verbose_llama,
    )


def build_engine(cfg: Config, backend, metrics):
    if cfg.engine == "simple":
        from .engine.simple import SimpleEngine

        return SimpleEngine(backend, cfg)
    if cfg.engine == "static":
        from .engine.static_batch import StaticBatchEngine

        return StaticBatchEngine(backend, cfg, metrics)
    from .engine.continuous import ContinuousBatchEngine

    return ContinuousBatchEngine(backend, cfg, metrics)


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        t0 = time.perf_counter()
        app.state.config = cfg
        app.state.metrics = MetricsRegistry()
        app.state.backend = build_backend(cfg)
        app.state.engine = build_engine(cfg, app.state.backend, app.state.metrics)
        await app.state.engine.start()
        app.state.started_t = time.time()
        app.state.load_time_s = time.perf_counter() - t0
        try:
            yield
        finally:
            await app.state.engine.stop()
            app.state.backend.close()

    app = FastAPI(title="llama-serve", version="0.7.0", lifespan=lifespan)

    # --- health / introspection ------------------------------------------
    @app.get("/health")
    async def health():
        b = app.state.backend
        return {
            "status": "ok",
            "backend": b.name,
            "engine": app.state.engine.name,
            "model": cfg.model_path if b.name != "mock" else "mock",
            "n_ctx": b.n_ctx,
            "n_ctx_per_seq": cfg.n_ctx_per_seq,
            "max_seqs": b.max_seqs,
            "load_time_s": round(app.state.load_time_s, 3),
            "uptime_s": round(time.time() - app.state.started_t, 1),
        }

    @app.get("/config")
    async def config():
        return cfg.as_dict()

    @app.get("/metrics")
    async def metrics():
        return JSONResponse(app.state.metrics.snapshot(app.state.engine))

    @app.post("/metrics/reset")
    async def metrics_reset():
        app.state.metrics.reset()
        return {"status": "reset"}

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        from .dashboard import DASHBOARD_HTML

        return HTMLResponse(DASHBOARD_HTML)

    # --- generation -------------------------------------------------------
    @app.post("/generate")
    async def generate(body: GenerateRequest, http_req: HTTPRequest):
        backend = app.state.backend
        engine = app.state.engine

        if body.messages:
            prompt = backend.format_chat([m.model_dump() for m in body.messages])
        elif body.prompt is not None:
            prompt = body.prompt
        else:
            raise HTTPException(422, "one of 'prompt' or 'messages' is required")

        req = Request(prompt=prompt, params=body.sampling(), priority=body.priority)
        try:
            engine.submit(req)
        except QueueFull as e:
            raise HTTPException(503, str(e)) from e
        except ContextOverflow as e:
            raise HTTPException(413, str(e)) from e

        app.state.metrics.on_arrival(req)

        if body.stream:
            return StreamingResponse(
                _sse(engine, req, app.state.metrics),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Request-Id": str(req.rid)},
            )

        async for _ in engine.stream(req):
            pass
        app.state.metrics.on_finish(req)
        if req.error:
            raise HTTPException(500, req.error)
        return {
            "id": req.rid,
            "text": req.text,
            "finish_reason": req.finish_reason.value if req.finish_reason else None,
            "usage": {
                "prompt_tokens": req.metrics.prompt_tokens,
                "cached_prompt_tokens": req.metrics.cached_prompt_tokens,
                "completion_tokens": req.metrics.generated_tokens,
                "total_tokens": req.metrics.prompt_tokens + req.metrics.generated_tokens,
            },
            "timings": req.metrics.as_dict(),
        }

    return app


async def _sse(engine, req: Request, metrics):
    """Server-sent events. Each token is flushed as it is produced, which is
    what makes client-side TTFT measurable rather than inferred."""
    try:
        async for piece in engine.stream(req):
            yield f"data: {json.dumps({'text': piece})}\n\n"
        metrics.on_finish(req)
        final = {
            "done": True,
            "finish_reason": req.finish_reason.value if req.finish_reason else None,
            "timings": req.metrics.as_dict(),
        }
        if req.error:
            final["error"] = req.error
        yield f"data: {json.dumps(final)}\n\n"
    except Exception as e:  # pragma: no cover
        yield f"data: {json.dumps({'done': True, 'error': str(e)})}\n\n"
    yield "data: [DONE]\n\n"


app = None  # populated by server.py / uvicorn factory
