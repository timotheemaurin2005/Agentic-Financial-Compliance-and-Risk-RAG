from rag_agent.graph import app

# Comparison query
result = app.invoke({
    "query": "How did the FOMC's inflation language change between the September 2024 and January 2025 statements?",
    "retry_count": 0, "retrieved_chunks": [], "table_chunks": [], "text_chunks": [], "cited_sources": []
})
print(result["final_answer"])
print(f"Contradiction detected: {result['contradiction_detected']}")

# Statement vs Minutes query
result2 = app.invoke({
    "query": "Did the January 2025 minutes reveal any disagreement that the statement didn't mention?",
    "retry_count": 0, "retrieved_chunks": [], "table_chunks": [], "text_chunks": [], "cited_sources": []
})
print(result2["final_answer"])
print(f"Contradiction detected: {result2['contradiction_detected']}")