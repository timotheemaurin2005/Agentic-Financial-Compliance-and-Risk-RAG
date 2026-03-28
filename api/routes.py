import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pinecone import Pinecone

from rag_agent.graph import run_query

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    contradiction_detected: Optional[bool] = None
    confidence: Optional[float] = None
    error: Optional[str] = None

@router.post("/query", response_model=QueryResponse)
async def query_sync(request: QueryRequest):
    """Synchronous endpoint that runs the query through the full LangGraph agent."""
    try:
        # run_query is a synchronous helper in graph.py
        result_state = run_query(query=request.question)
        
        if result_state.get("error"):
            raise HTTPException(status_code=500, detail=result_state["error"])
            
        return QueryResponse(
            answer=result_state.get("final_answer"),
            sources=result_state.get("cited_sources", []),
            contradiction_detected=result_state.get("contradiction_detected"),
            confidence=result_state.get("confidence_score")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
async def get_documents():
    """Returns a list of ingested documents from Pinecone's index stats."""
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "fin-compliance-rag")
    
    if not pinecone_api_key:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY environment variable not set")
        
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        return {
            "total_vector_count": stats.get("total_vector_count"),
            "namespaces": stats.get("namespaces", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
