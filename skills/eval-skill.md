# Evaluation Skill — RAGAS & Ablation Testing

## Purpose

This skill governs how the RAG system is evaluated. Every claim about system performance must be backed by quantitative metrics from this framework. Building the system is half the work — proving it works is what makes this portfolio-ready.

## Evaluation Framework: RAGAS

Use the `ragas` Python library (v0.1+). Four core metrics:

| Metric              | What It Measures                                        | Target  |
|----------------------|--------------------------------------------------------|---------|
| **Faithfulness**     | Is the answer grounded in retrieved context?            | >= 0.85 |
| **Answer Relevancy** | Does the answer address the actual question?            | >= 0.80 |
| **Context Precision**| Are the retrieved chunks relevant (not noise)?          | >= 0.75 |
| **Context Recall**   | Did retrieval find ALL relevant chunks?                 | >= 0.70 |

### RAGAS Setup

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

eval_data = {
    "question": [...],           # 30 questions
    "answer": [...],             # System-generated answers
    "contexts": [...],           # List[List[str]] — retrieved chunks per question
    "ground_truth": [...],       # Human-written reference answers
}

dataset = Dataset.from_dict(eval_data)
results = evaluate(dataset, metrics=metrics)
```

## Eval Dataset Specification

### Format (`eval/data/eval_set.json`)

```json
[
  {
    "id": "q001",
    "question": "What was the target federal funds rate after the January 2025 FOMC meeting?",
    "query_type": "factual",
    "ground_truth": "The FOMC maintained the target range at 4-1/4 to 4-1/2 percent at the January 2025 meeting.",
    "source_docs": ["fomc_statement_2025-01-29"],
    "expected_sections": ["rate_decision"],
    "difficulty": "easy"
  },
  {
    "id": "q018",
    "question": "How did the FOMC's forward guidance language change between the September 2024 and January 2025 statements?",
    "query_type": "contradiction",
    "ground_truth": "In September 2024, the Committee initiated rate cuts with a 50bp reduction, signaling confidence that inflation was moving toward target. By January 2025, the Committee held rates steady and removed language about gaining 'greater confidence' on inflation, instead noting that inflation 'remains somewhat elevated'.",
    "source_docs": ["fomc_statement_2024-09-18", "fomc_statement_2025-01-29"],
    "expected_sections": ["forward_guidance", "rate_decision"],
    "difficulty": "hard"
  }
]
```

### Distribution Requirements

Build exactly **30 questions** with this distribution:

| Query Type     | Count | Difficulty Split        | Example Focus                                  |
|----------------|-------|--------------------------|------------------------------------------------|
| Factual        | 8     | 5 easy, 3 medium         | Rate decisions, vote tallies, specific language |
| Numerical      | 8     | 3 easy, 3 medium, 2 hard | Rate changes, vote counts, basis point moves    |
| Comparison     | 7     | 2 easy, 3 medium, 2 hard | Statement vs minutes, language evolution         |
| Contradiction  | 7     | 1 easy, 3 medium, 3 hard | Guidance shifts, risk assessment changes         |

### Example Questions by Type

**Factual:**
- "Who voted against the rate decision at the [date] meeting?"
- "What did the FOMC say about the labor market in the [date] statement?"
- "What was the staff's GDP growth forecast in the [date] minutes?"

**Numerical:**
- "By how many basis points did the FOMC cut rates in September 2024?"
- "How many members dissented at the [date] meeting?"
- "What target range was set at the December 2024 meeting?"

**Comparison:**
- "How did the statement's economic assessment differ from the staff outlook in the minutes for the [date] meeting?"
- "Compare the risk assessment language between the [date1] and [date2] minutes"
- "Did the statement and minutes align on the degree of consensus at the [date] meeting?"

**Contradiction:**
- "How did inflation language change between the September and December 2024 statements?"
- "The January 2025 statement suggests consensus, but do the minutes reveal disagreement?"
- "Track the evolution of forward guidance language across all 5 meetings"

### Difficulty Definitions

- **Easy:** Answer is in a single chunk from a single document
- **Medium:** Requires synthesising 2–3 chunks, may span sections within one document
- **Hard:** Requires cross-document reasoning, comparing subtle language shifts across meetings, or reconciling statement vs minutes

### Writing Good Ground Truths

- Be specific: include exact rate figures, exact Fed language in quotes, and meeting dates
- For contradiction questions: state BOTH positions with their meeting dates and quote the specific language that changed
- For numerical questions: include the calculation, not just the result
- Keep ground truths to 1–3 sentences
- Use exact Fed terminology (e.g., "participants" not "members", "Committee" not "Fed")

## Ablation Study Specification

### Variables to Sweep

Run the full eval set against each configuration. Change ONE variable at a time.

| Variable            | Values to Test                                | Default        |
|---------------------|-----------------------------------------------|----------------|
| Chunk size (tokens) | 256, 512, 768, 1024                           | 512            |
| Chunk overlap       | 0, 25, 50, 100 tokens                         | 50             |
| top_k retrieval     | 5, 10, 15, 20                                 | 10             |
| Embedding model     | text-embedding-3-small, text-embedding-3-large| 3-large        |
| With/without summary embedding | True, False                       | True           |
| LLM for synthesis   | gpt-4o, gpt-4o-mini, claude-sonnet            | gpt-4o         |

### Results Logging

Store every experiment run in a structured JSON log:

```json
{
  "experiment_id": "exp_007",
  "timestamp": "2026-03-28T14:30:00Z",
  "config": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 10,
    "embedding_model": "text-embedding-3-large",
    "use_summary_embedding": true,
    "llm_model": "gpt-4o"
  },
  "results": {
    "faithfulness": 0.89,
    "answer_relevancy": 0.84,
    "context_precision": 0.78,
    "context_recall": 0.73,
    "mean_latency_seconds": 4.2,
    "total_cost_usd": 0.85
  },
  "per_query_results": [
    {
      "id": "q001",
      "query_type": "factual",
      "difficulty": "easy",
      "faithfulness": 0.95,
      "answer_relevancy": 0.92,
      "context_precision": 0.90,
      "context_recall": 0.85
    }
  ]
}
```

### Ablation Runner (`eval/ablation.py`)

```python
import itertools
import json
from datetime import datetime

SWEEP_CONFIGS = {
    "chunk_size": [256, 512, 768, 1024],
    "top_k": [5, 10, 15, 20],
}

def run_ablation(variable: str, values: list, base_config: dict):
    """Sweep one variable while holding others at default."""
    results = []
    for val in values:
        config = {**base_config, variable: val}
        experiment = {
            "experiment_id": f"exp_{variable}_{val}",
            "timestamp": datetime.utcnow().isoformat(),
            "config": config,
            "results": run_ragas(config),
        }
        results.append(experiment)
    return results
```

## Reporting

### Summary Table (for README)

After running all ablations, produce a summary table:

```markdown
| Config                  | Faithfulness | Relevancy | Precision | Recall | Latency (s) |
|-------------------------|-------------|-----------|-----------|--------|-------------|
| Baseline (512/10/3-lg)  | 0.89        | 0.84      | 0.78      | 0.73   | 4.2         |
| chunk_size=256          | 0.91        | 0.82      | 0.80      | 0.68   | 3.8         |
| chunk_size=1024         | 0.85        | 0.86      | 0.72      | 0.77   | 5.1         |
| top_k=5                 | 0.92        | 0.81      | 0.85      | 0.61   | 3.1         |
| top_k=20                | 0.84        | 0.85      | 0.70      | 0.82   | 6.3         |
```

### Key Findings Section

Write a short analysis (3–5 bullet points) interpreting the results:
- Which config maximises faithfulness? (Most important for policy analysis)
- What is the faithfulness-recall tradeoff as top_k increases?
- Does dual embedding justify its cost?
- Where does the system fail? (Analyse the lowest-scoring questions — likely the subtle language shift contradictions)

## Error Analysis

For any question scoring below 0.6 on any metric:
1. Log the question, retrieved chunks, generated answer, and ground truth
2. Classify the failure mode:
   - **Retrieval miss:** Relevant chunk not retrieved (context recall issue)
   - **Noise retrieval:** Irrelevant chunks diluting context (context precision issue)
   - **Hallucination:** Answer contains claims not in context (faithfulness issue)
   - **Language nuance miss:** System failed to detect subtle Fed language shift (domain-specific failure)
   - **Cross-meeting confusion:** System mixed up which meeting said what (metadata filtering issue)
   - **Partial answer:** Answer is correct but incomplete (recall issue)
3. Suggest a fix for each failure mode

## Testing Checklist

- [ ] Eval dataset has exactly 30 questions with correct distribution
- [ ] RAGAS runs without errors on the full dataset
- [ ] All 4 metrics are computed and logged
- [ ] At least one ablation sweep is complete (e.g., chunk_size)
- [ ] Results JSON is valid and includes per-query breakdown
- [ ] Summary table is generated for README
- [ ] Error analysis covers all questions scoring below 0.6