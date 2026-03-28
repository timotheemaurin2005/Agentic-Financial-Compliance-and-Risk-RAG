import json
import logging
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_agent.graph import app
from rag_agent.state import RAGState

router = APIRouter()
logger = logging.getLogger(__name__)

class StreamQueryRequest(BaseModel):
    question: str

async def generate_sse_events(query: str):
    """Generator for Server-Sent Events (SSE) from the LangGraph run."""
    initial_state: RAGState = {
        "query": query,
        "query_type": None,
        "metadata_filters": None,
        "retrieved_chunks": [],
        "table_chunks": [],
        "text_chunks": [],
        "draft_answer": None,
        "cited_sources": [],
        "contradiction_detected": None,
        "is_grounded": None,
        "confidence_score": None,
        "retry_count": 0,
        "final_answer": None,
        "error": None,
    }

    status_mapping = {
        "router": "Routing query...",
        "retriever": "Retrieving documents...",
        "table_reasoner": "Reasoning over tables...",
        "synthesizer": "Synthesizing answer...",
        "verifier": "Verifying grounding..."
    }

    try:
        # We use astream_events to get both node transitions and LLM tokens
        async for event in app.astream_events(initial_state, version="v2"):
            kind = event["event"]
            name = event.get("name", "")
            
            # Streaming node start events as status updates
            if kind == "on_chain_start" and name in status_mapping:
                yield f"data: {json.dumps({'type': 'status', 'status': status_mapping[name]})}\n\n"
                
            # Streaming LLM tokens during synthesis
            elif kind == "on_chat_model_stream":
                # Assuming the token is in chunk.content
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk.content})}\n\n"
            
            # Emitting final answer and metadata once the main graph concludes
            elif kind == "on_chain_end" and name == "LangGraph":
                output_state = event.get("data", {}).get("output", {})
                
                final_data = {
                    "type": "final",
                    "answer": output_state.get("final_answer"),
                    "sources": output_state.get("cited_sources", []),
                    "contradiction_detected": output_state.get("contradiction_detected"),
                    "confidence": output_state.get("confidence_score")
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                
    except Exception as e:
        logger.error(f"Error in graph streaming: {e}")
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    yield "data: [DONE]\n\n"

@router.post("/query")
async def stream_query(request: StreamQueryRequest):
    """Streaming endpoint that yields status updates and tokens."""
    return StreamingResponse(
        generate_sse_events(request.question), 
        media_type="text/event-stream"
    )
