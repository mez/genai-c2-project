"""
batch_evaluate.py

Batch evaluation script for NASA RAG project.
Loads evaluation_dataset.txt, runs retrieval + answer generation + evaluation for each question.
Outputs per-question metrics and aggregate summaries.
"""
import re
import sys
import argparse
import statistics
from pathlib import Path
from typing import List, Dict

from rag_client import discover_chroma_backends, initialize_rag_system, retrieve_documents, format_context
from llm_client import generate_response
from ragas_evaluator import evaluate_response_quality

EVAL_FILE = "evaluation_dataset.txt"

# Parse evaluation_dataset.txt into list of dicts
QUESTION_RE = re.compile(r"Q: (.*?)\nA: (.*?)(?:\n\d+\.|\Z)", re.DOTALL)
def load_eval_dataset(path: str) -> List[Dict]:
    text = Path(path).read_text()
    items = []
    for match in QUESTION_RE.finditer(text):
        q = match.group(1).strip().replace('\n', ' ')
        a = match.group(2).strip().replace('\n', ' ')
        items.append({"question": q, "reference_answer": a})
    return items

def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for NASA RAG project.")
    parser.add_argument('--eval-file', type=str, default=EVAL_FILE, help="Evaluation dataset file")
    parser.add_argument('--collection', type=str, default=None, help="ChromaDB collection (auto if not set)")
    parser.add_argument('--n-docs', type=int, default=3, help="Number of docs to retrieve")
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo", help="LLM model name")
    args = parser.parse_args()

    # Load evaluation dataset
    eval_items = load_eval_dataset(args.eval_file)
    if not eval_items:
        print(f"No questions found in {args.eval_file}")
        sys.exit(1)

    # Discover ChromaDB backends
    backends = discover_chroma_backends()
    if not backends:
        print("No ChromaDB backends found. Run embedding pipeline first.")
        sys.exit(1)
    # Pick first collection if not specified
    if args.collection is None:
        first_key = next(iter(backends))
        chroma_dir = backends[first_key]["directory"]
        collection_name = backends[first_key]["collection_name"]
    else:
        # Find collection by name
        found = False
        for b in backends.values():
            if b["collection_name"] == args.collection:
                chroma_dir = b["directory"]
                collection_name = b["collection_name"]
                found = True
                break
        if not found:
            print(f"Collection {args.collection} not found.")
            sys.exit(1)

    collection, ok, err = initialize_rag_system(chroma_dir, collection_name)
    if not ok:
        print(f"Failed to initialize RAG system: {err}")
        sys.exit(1)

    all_metrics = []
    print(f"Evaluating {len(eval_items)} questions...")
    for idx, item in enumerate(eval_items, 1):
        q = item["question"]
        ref = item["reference_answer"]
        # Retrieve docs
        docs_result = retrieve_documents(collection, q, args.n_docs)
        if docs_result and docs_result.get("documents"):
            context = format_context(docs_result["documents"][0], docs_result["metadatas"][0])
            contexts_list = docs_result["documents"][0]
        else:
            context = ""
            contexts_list = []
        # Generate answer
        answer = generate_response(q, context, [], args.model)
        # Evaluate
        metrics = evaluate_response_quality(q, answer, contexts_list)
        all_metrics.append(metrics)
        print(f"Q{idx}: {q}\n  Reference: {ref}\n  Answer: {answer}")
        print(f"  Metrics: {metrics}\n")

    # Aggregate metrics
    if all_metrics:
        metric_names = all_metrics[0].keys()
        print("Aggregate metric summary:")
        for m in metric_names:
            vals = [x[m] for x in all_metrics if m in x and isinstance(x[m], (int, float))]
            if vals:
                print(f"  {m}: mean={statistics.mean(vals):.3f} std={statistics.stdev(vals) if len(vals)>1 else 0:.3f} min={min(vals):.3f} max={max(vals):.3f}")

if __name__ == "__main__":
    main()
