"""
app.py — Minimal REST API exposing the Resume Intelligence model.

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000

Run in Colab (behind a tunnel, e.g. ngrok/cloudflared) or just call
`parse_resume()` directly from resume_parser.py without the API layer
if you only need programmatic access from within the same notebook.
"""
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import config
from resume_parser import get_parser, ModelLoadError, ParseError

logging.basicConfig(level=config.log_level)
logger = logging.getLogger("app")

app = FastAPI(
    title="Resume Intelligence API",
    description="Extracts structured JSON from raw resume text using a fine-tuned open-source LLM.",
    version="1.0.0",
)


class ParseRequest(BaseModel):
    resume_text: str = Field(..., description="Raw resume text (from PDF extraction, paste, etc.)")


class ParseResponse(BaseModel):
    result: dict
    processing_time_seconds: float


@app.on_event("startup")
def load_model_on_startup():
    """Load the model once when the server starts, not per-request."""
    try:
        get_parser()
    except ModelLoadError as e:
        # Log but don't crash the process — /health will report unhealthy,
        # and /parse will return a clear 503 rather than a bare stack trace.
        logger.error(f"Model failed to load at startup: {e}")


@app.get("/health")
def health():
    parser = None
    try:
        parser = get_parser()
        loaded = parser._loaded
    except Exception:
        loaded = False
    return {"status": "ok" if loaded else "model_unavailable", "model_loaded": loaded}


@app.post("/parse", response_model=ParseResponse)
def parse_resume(req: ParseRequest):
    if not req.resume_text or not req.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text must not be empty.")

    try:
        parser = get_parser()
    except ModelLoadError as e:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e}")

    start = time.time()
    try:
        result = parser.parse_resume(req.resume_text)
    except ParseError as e:
        raise HTTPException(status_code=422, detail=f"Could not parse resume: {e}")
    except Exception as e:
        logger.exception("Unexpected error during parsing")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return ParseResponse(result=result, processing_time_seconds=round(time.time() - start, 3))


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.host, port=config.port, reload=False)
