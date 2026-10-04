"""Paragraph chunking + dense-only baseline, using the same answer generator as production."""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EMBEDDING_MODEL, NAIVE_COLLECTION
from src.generation import generate_answer
from src.llm import get_settings
from src.m1_chunking import chunk_basic, load_documents
from src.m2_search import DenseSearch
from src.m4_eval import METRICS, evaluate_ragas, load_test_set, save_report


def main():
    started = time.perf_counter()
    documents = load_documents()
    chunks = [
        {"text": c.text, "metadata": c.metadata}
        for doc in documents
        for c in chunk_basic(doc["text"], metadata=doc["metadata"])
    ]
    chunk_ms = (time.perf_counter() - started) * 1000
    print(f"BASELINE: {len(chunks)} paragraph chunks", flush=True)
    started = time.perf_counter()
    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)
    index_ms = (time.perf_counter() - started) * 1000
    questions, answers, contexts, truths, traces = [], [], [], [], []
    for i, item in enumerate(load_test_set(), 1):
        started = time.perf_counter()
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        retrieval_ms = (time.perf_counter() - started) * 1000
        evidence = [r.text for r in results]
        started = time.perf_counter()
        answer, backend = generate_answer(item["question"], evidence)
        traces.append(
            {
                "question": item["question"],
                "retrieval_ms": retrieval_ms,
                "generation_ms": (time.perf_counter() - started) * 1000,
                "generation_backend": backend,
                "sources": [r.metadata.get("source") for r in results],
            }
        )
        questions.append(item["question"])
        answers.append(answer)
        contexts.append(evidence)
        truths.append(item["ground_truth"])
        print(f"  [{i}] {item['question']}", flush=True)
    started = time.perf_counter()
    results = evaluate_ragas(questions, answers, contexts, truths)
    results["latency"] = {
        "build_ms": {"load_and_chunk_ms": chunk_ms, "index_ms": index_ms},
        "per_query": traces,
        "evaluation_ms": (time.perf_counter() - started) * 1000,
    }
    results["configuration"] = {
        "llm_provider": get_settings().provider,
        "llm_model": get_settings().model,
        "embedding_model": EMBEDDING_MODEL,
        "documents": len(documents),
        "chunks": len(chunks),
        "search": "dense_only",
    }
    for metric in METRICS:
        value = f"{results[metric]:.4f}" if results["metric_sample_counts"][metric] else "N/A (not evaluated)"
        print(f"  {metric}: {value}")
    save_report(results, [], path="reports/naive_baseline_report.json")
    return results


if __name__ == "__main__":
    main()
