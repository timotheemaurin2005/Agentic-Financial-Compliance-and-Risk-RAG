"""Table reasoning prompt for structured table analysis."""

TABLE_REASONING_PROMPT = """\
You are a monetary policy analyst. Analyse the following tables extracted from \
FOMC documents.

{formatted_tables}

Question: {query}

Provide:
1. Key data points and their values
2. Changes between meetings (if multiple periods present)
3. Any notable shifts (e.g., new dissenting votes, changed rate targets)
4. Whether the tables support or contradict each other

Be precise. Cite which meeting date each figure comes from."""
