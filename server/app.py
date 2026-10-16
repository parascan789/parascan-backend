"""PARASCAN inference API.

Run:
    uv run python -m server [--host 0.0.0.0] [--port 8000] [--reload]

Environment:
    PARASCAN_DEVICE        torch device, e.g. "cpu" or "cuda" (default: cuda if available)
    PARASCAN_CORS_ORIGINS  comma-separated allowed origins (default: none)
    PARASCAN_MAX_UPLOAD_MB max image size in MB (default: 15)
    PARASCAN_DISEASE_THRESHOLD  minimum disease probability before classifying species (default: 0.80)
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from parascan.pipeline import Pipeline

MAX_UPLOAD_BYTES = int(float(os.environ.get("PARASCAN_MAX_UPLOAD_MB", "15")) * 1024 * 1024)
pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipeline
    pipeline = Pipeline(os.environ.get("PARASCAN_DEVICE"))
    yield


app = FastAPI(
    title="PARASCAN",
    description="Blood smear classification: smear type, then disease (malaria / lymphatic filariasis), then species.",
    version="0.1.0",
    lifespan=lifespan,
)

origins = [o.strip() for o in os.environ.get("PARASCAN_CORS_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if pipeline else "loading"}


@app.get("/models")
def models() -> dict:
    return pipeline.info()


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="Blood smear image (JPEG/PNG/...)")) -> dict:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        result = await run_in_threadpool(pipeline.predict_bytes, data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "File is not a readable image.")
    return {"filename": file.filename, **result}
