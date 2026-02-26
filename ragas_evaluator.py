from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
    # Import correct metrics from ragas.metrics
from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
        answer_similarity,
    )

from utils import get_openai_api_key

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # Instantiate LLM and embeddings wrappers (no keyword args)
    llm = LangchainLLMWrapper(ChatOpenAI(
        api_key=get_openai_api_key(),
        base_url="https://openai.vocareum.com/v1",
        model="gpt-3.5-turbo", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(
        api_key=get_openai_api_key(),
        base_url="https://openai.vocareum.com/v1",
        model="text-embedding-3-small"))

    # Use alternative metrics as requested
    metrics = [
        BleuScore(),
        NonLLMContextPrecisionWithReference(),
        ResponseRelevancy(),
        Faithfulness(),
        RougeScore(),
    ]

    # Prepare data as DataFrame for ragas
    import pandas as pd
    from datasets import Dataset
    data = pd.DataFrame([
        {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "reference": answer,
            "reference_contexts": contexts,
        }
    ])
    dataset = Dataset.from_pandas(data)

    # Evaluate
    results = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )

    # Extract scores for each metric robustly
    scores = {}
    for metric in metrics:
        name = metric.name
        try:
            val = results[name]
            # Handle list or scalar
            if isinstance(val, list) and val:
                scores[name] = float(val[0])
            elif isinstance(val, (int, float)):
                scores[name] = float(val)
            else:
                scores[name] = float(val) if val is not None else None
        except (KeyError, Exception):
            scores[name] = None
    return scores
