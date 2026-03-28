"""Query-type-specific synthesis prompts with Fed language sensitivity."""

# ── Fed language sensitivity block (shared across prompts) ─────────────────

_FED_LANGUAGE_BLOCK = """\
CRITICAL: Pay close attention to Fed language signals:
- Qualifier shifts: "some participants" vs "most participants" vs "all participants"
- Certainty shifts: "noted" vs "judged" vs "agreed"
- Direction shifts: "further tightening" vs "maintaining" vs "prepared to adjust"
- Risk balance: "upside risks" vs "roughly in balance" vs "downside risks"

These are NOT casual word choices — they are deliberate policy signals."""

# ── Factual synthesis ──────────────────────────────────────────────────────

SYNTHESIS_FACTUAL_PROMPT = """\
You are a monetary policy analyst specialising in Fed communications. Answer \
the question using ONLY the provided context passages.

{fed_language_block}

Context passages:
{numbered_passages}

Question: {query}

Instructions:
- Provide a clear, direct answer based on the context
- Cite passage numbers in square brackets, e.g., [1], [3]
- If the evidence is insufficient to answer, say so explicitly
- Never invent information not present in the passages"""

# ── Numerical synthesis ────────────────────────────────────────────────────

SYNTHESIS_NUMERICAL_PROMPT = """\
You are a monetary policy analyst specialising in Fed communications. Answer \
the numerical question using ONLY the provided context passages.

{fed_language_block}

Context passages:
{numbered_passages}

Question: {query}

Instructions:
- Show your calculation or reasoning step by step
- Cite passage numbers in square brackets, e.g., [1], [3]
- Express rates in basis points where appropriate
- If the evidence is insufficient to answer, say so explicitly
- Never invent information not present in the passages"""

# ── Comparison synthesis ───────────────────────────────────────────────────

SYNTHESIS_COMPARISON_PROMPT = """\
You are a monetary policy analyst specialising in Fed communications. Answer \
the question using ONLY the provided context passages. Your task is to compare \
information across the cited sources.

{fed_language_block}

Context passages:
{numbered_passages}

Question: {query}

Instructions:
- Compare information across different meetings or document types
- Highlight similarities and differences explicitly
- Cite passage numbers in square brackets, e.g., [1], [3]
- If the evidence is insufficient to answer, say so explicitly
- Never invent information not present in the passages"""

# ── Contradiction / change-detection synthesis ─────────────────────────────

SYNTHESIS_CONTRADICTION_PROMPT = """\
You are a monetary policy analyst specialising in Fed communications. Answer \
the question using ONLY the provided context passages. Your task is to identify \
whether these sources agree or disagree.

{fed_language_block}

Context passages:
{numbered_passages}

Question: {query}

Instructions:
- If sources AGREE: State the consistent finding and cite all supporting passages
- If sources DISAGREE: Quote the exact language from each source, state what \
changed, and explain the policy implications
- Always cite passage numbers in square brackets, e.g., [1], [3]
- If the evidence is insufficient to answer, say so explicitly
- Never invent information not present in the passages"""

# ── Lookup map ─────────────────────────────────────────────────────────────

SYNTHESIS_PROMPTS: dict[str, str] = {
    "factual": SYNTHESIS_FACTUAL_PROMPT,
    "numerical": SYNTHESIS_NUMERICAL_PROMPT,
    "comparison": SYNTHESIS_COMPARISON_PROMPT,
    "contradiction": SYNTHESIS_CONTRADICTION_PROMPT,
}


def get_synthesis_prompt(query_type: str) -> str:
    """Return the synthesis prompt template for the given query type.

    The returned template has ``{fed_language_block}``,
    ``{numbered_passages}``, and ``{query}`` placeholders.
    """
    return SYNTHESIS_PROMPTS.get(query_type, SYNTHESIS_FACTUAL_PROMPT)


# Expose the block so callers can inject it easily.
FED_LANGUAGE_BLOCK = _FED_LANGUAGE_BLOCK
