"""FastAPI application entry point."""
from fastapi import FastAPI

app = FastAPI(title="Financial Compliance RAG API", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
