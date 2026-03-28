"""FastAPI application entry point."""
from dotenv import load_dotenv

# Load environment variables before anything else
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

# Include routers
app.include_router(standard_router, tags=["standard"])
app.include_router(streaming_router, prefix="/stream", tags=["streaming"])

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
