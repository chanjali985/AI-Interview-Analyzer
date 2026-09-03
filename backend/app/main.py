"""FastAPI application factory."""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .database import SessionLocal, init_db
from .logging_config import configure_logging
from .routers import auth, health, interviews, public, roles
from .worker import requeue_unfinished, shutdown

logger = logging.getLogger(__name__)

FRONTEND_DIST = os.environ.get(
    "FRONTEND_DIST",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend", "dist"),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.LOG_LEVEL, json_logs=settings.is_production)
    logger.info("Starting %s in %s mode", settings.APP_NAME, settings.ENVIRONMENT)

    init_db()
    db = SessionLocal()
    try:
        from .seed import run as run_seed

        run_seed(db)
    finally:
        db.close()

    if settings.is_production and settings.ADMIN_PASSWORD == "change-me-now":
        logger.error("ADMIN_PASSWORD is still the default value. Set it before exposing this service.")

    try:
        requeue_unfinished()
    except Exception:  # noqa: BLE001
        logger.exception("Could not requeue unfinished interviews")

    yield

    shutdown(wait=False)
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=health.VERSION,
        description="Automated interview analysis: transcription, scoring, skill matching and emailed reports.",
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "Unhandled error on %s %s", request.method, request.url.path, extra={"request_id": request_id}
            )
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        if not request.url.path.startswith(("/assets", "/static")):
            logger.info(
                "%s %s -> %s in %.0fms",
                request.method,
                request.url.path,
                response.status_code,
                elapsed,
                extra={"request_id": request_id},
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(part) for part in first.get("loc", [])[1:]) or "request"
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": f"{field}: {first.get('msg', 'invalid value')}", "errors": exc.errors()[:5]},
        )

    api = settings.API_PREFIX
    app.include_router(health.router, prefix=api)
    app.include_router(auth.router, prefix=api)
    app.include_router(roles.router, prefix=api)
    app.include_router(interviews.router, prefix=api)
    app.include_router(public.router, prefix=api)

    @app.get(f"{api}/config", tags=["meta"])
    def public_config() -> dict:
        """Non-secret settings the frontend needs at boot."""
        return {
            "app_name": settings.APP_NAME,
            "company_name": settings.COMPANY_NAME,
            "environment": settings.ENVIRONMENT,
            "shortlist_threshold": settings.SHORTLIST_THRESHOLD,
            "max_audio_mb": settings.MAX_AUDIO_MB,
            "allowed_resume_extensions": settings.ALLOWED_RESUME_EXTENSIONS,
        }

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built React app, falling back to index.html for client routes."""
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if not os.path.isfile(index_path):
        logger.warning("Frontend build not found at %s — serving API only", FRONTEND_DIST)

        @app.get("/", include_in_schema=False)
        def api_only_root() -> dict:
            return {
                "app": settings.APP_NAME,
                "docs": "/docs" if not settings.is_production else None,
                "health": f"{settings.API_PREFIX}/health",
                "note": "Frontend has not been built. Run `npm run build` in ./frontend.",
            }

        return

    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith(settings.API_PREFIX.lstrip("/")) or full_path.startswith("api/"):
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        candidate = os.path.normpath(os.path.join(FRONTEND_DIST, full_path))
        if (
            full_path
            and candidate.startswith(os.path.abspath(FRONTEND_DIST))
            and os.path.isfile(candidate)
        ):
            return FileResponse(candidate)
        return FileResponse(index_path)


app = create_app()
