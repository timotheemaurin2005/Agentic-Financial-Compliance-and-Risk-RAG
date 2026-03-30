"""FastAPI application entry point."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables before anything else
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from api.routes import router as standard_router
from api.streaming import router as streaming_router

app = FastAPI(title="Financial Compliance RAG API", version="0.1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers (must be before static file mount)
app.include_router(standard_router, prefix="/api", tags=["standard"])
app.include_router(streaming_router, prefix="/api/stream", tags=["streaming"])

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

# ── Serve Frontend ──
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

@app.get("/analyst", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the SPA index.html for both / and /analyst routes."""
    index_path = FRONTEND_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

# ── Serve Raw Documents ──
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
app.mount("/raw", StaticFiles(directory=str(RAW_DATA_DIR)), name="raw_data")

# Mount static assets at root (after explicit routes so they don't override them)
# This serves styles.css, app.js etc. at their relative paths
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
