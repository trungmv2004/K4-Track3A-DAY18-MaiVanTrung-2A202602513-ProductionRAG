from __future__ import annotations

"""Production RAG: hierarchical retrieval, enrichment, hybrid search and reranking."""

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBEDDING_MODEL, RERANK_TOP_K
from src.generation import SYSTEM_PROMPT, generate_answer
from src.llm import get_settings
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import METRICS, evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import enrich_chunks


def _document_metadata(documents):
    """Infer policy families from titles, with explicit version/effective-date evidence."""
    latest = {}
    for document in documents:
        text, metadata = document["text"], document["metadata"]
        title = re.search(r"^#\s+(.+)", text, re.MULTILINE)
        family = re.sub(r"\s*\([^)]*\)\s*$", "", title.group(1)).strip() if title else metadata["source"]
        metadata["title"] = title.group(1) if title else metadata["source"]
        version = re.search(r"Phiên bản:\s*([\d.]+)", text)
        date = re.search(r"Ngày hiệu lực:\s*(\d{2}/\d{2}/\d{4})", text)
        metadata["policy_family"] = family
        metadata["version"] = version.group(1) if version else ""
        metadata["effective_date"] = date.group(1) if date else ""
        metadata["superseded"] = bool(re.search(r"đã (?:được )?thay thế", text, re.IGNORECASE))
        if version and date:
            key = (datetime.strptime(date.group(1), "%d/%m/%Y"), tuple(map(int, version.group(1).split("."))))
            metadata["_version_key"] = key
            if key[0].date() <= datetime.now(ZoneInfo("Asia/Bangkok")).date():
                latest[family] = max(latest.get(family, key), key)
    for document in documents:
        metadata = document["metadata"]
        key = metadata.pop("_version_key", None)
        if key and key < latest.get(metadata["policy_family"], key):
            metadata["superseded"] = True


def _validate_enrichment_cache(chunks, indexed_cache):
    def identity(text, metadata):
        return text, metadata.get("source"), metadata.get("parent_id")

    expected = Counter(identity(c["text"], c["metadata"]) for c in chunks)
    actual = Counter(identity(c.get("metadata", {}).get("original_text"), c.get("metadata", {}))
                     for c in indexed_cache)
    if expected != actual:
        raise ValueError("Enrichment cache does not match current source chunks")
    buckets = defaultdict(list)
    for cached in indexed_cache:
        if not isinstance(cached.get("text"), str):
            raise ValueError("Enrichment cache must contain indexed text")
        buckets[identity(cached["metadata"]["original_text"], cached["metadata"])].append(cached)
    ordered = []
    for chunk in chunks:
        cached = buckets[identity(chunk["text"], chunk["metadata"])].pop()
        if any(cached["metadata"].get(k) != v for k, v in chunk["metadata"].items()):
            raise ValueError("Enrichment cache has stale source metadata")
        ordered.append(cached)
    return ordered


def build_pipeline(indexed_cache=None):
    timings = {}
    start = time.perf_counter()
    documents = load_documents()
    _document_metadata(documents)
    parents_by_id, chunks = {}, []
    for document in documents:
        parents, children = chunk_hierarchical(document["text"], metadata=document["metadata"])
        parents_by_id.update({p.metadata["parent_id"]: p for p in parents})
        chunks.extend(
            {"text": c.text, "metadata": {**c.metadata, "parent_id": c.parent_id}} for c in children
        )
    timings["load_and_chunk_ms"] = (time.perf_counter() - start) * 1000
    print(
        f"[M1] {len(documents)} documents → {len(parents_by_id)} parents, {len(chunks)} children", flush=True
    )

    start = time.perf_counter()
    if indexed_cache is None:
        enriched = enrich_chunks(chunks)
        indexed = [
            {"text": e.enriched_text, "metadata": {**e.auto_metadata, "original_text": e.original_text}}
            for e in enriched
        ]
    else:
        indexed = _validate_enrichment_cache(chunks, indexed_cache)
        print(f"[M5] Reused {len(indexed)} previously enriched chunks; no enrichment API calls.", flush=True)
    timings["enrichment_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    search = HybridSearch()
    search.index(indexed)
    search.parents_by_id = parents_by_id
    timings["index_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    reranker = CrossEncoderReranker()
    reranker._load_model()
    timings["reranker_load_ms"] = (time.perf_counter() - start) * 1000
    search.build_timings = timings
    search.document_count = len(documents)
    search.chunk_count = len(chunks)
    search.enrichment_backends = {}
    for chunk in indexed:
        backend = chunk["metadata"].get("enrichment_backend", "unknown")
        search.enrichment_backends[backend] = search.enrichment_backends.get(backend, 0) + 1
    search.enrichment_cache_reused = indexed_cache is not None
    settings = get_settings()
    search.pipeline_fingerprint = hashlib.sha256(json.dumps({
        "indexed": indexed, "provider": settings.provider, "model": settings.model,
        "embedding_model": EMBEDDING_MODEL, "reranker_model": reranker.model_name,
        "answer_prompt": SYSTEM_PROMPT, "top_k": RERANK_TOP_K,
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    Path(".cache").mkdir(exist_ok=True)
    Path(".cache/production_enriched_index.json").write_text(
        json.dumps(indexed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[M2/M3/M5] Index and reranker ready.", flush=True)
    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    start = time.perf_counter()
    results = search.search(query)
    retrieval_ms = (time.perf_counter() - start) * 1000
    historical = bool(
        re.search(r"\b20\d{2}\b|phiên bản cũ|chính sách cũ|\bv1(?:\.0)?\b", query, re.IGNORECASE)
    )
    candidates = [r for r in results if historical or not r.metadata.get("superseded", False)]
    docs = [
        {"text": r.metadata.get("original_text", r.text), "score": r.score, "metadata": r.metadata}
        for r in candidates
    ]
    start = time.perf_counter()
    # Select distinct parents after ranking, so siblings do not consume the context budget.
    ranked = reranker.rerank(query, docs, top_k=len(docs))
    rerank_ms = (time.perf_counter() - start) * 1000
    contexts, selected, seen = [], [], set()
    for result in ranked:
        pid = result.metadata.get("parent_id")
        identity = pid or result.text
        if identity in seen:
            continue
        seen.add(identity)
        parent = getattr(search, "parents_by_id", {}).get(pid)
        context = parent.text if parent else result.text
        contexts.append(context)
        selected.append(
            {
                "source": result.metadata.get("source"),
                "parent_id": pid,
                "version": result.metadata.get("version"),
                "rerank_score": result.rerank_score,
            }
        )
        if len(contexts) == RERANK_TOP_K:
            break
    start = time.perf_counter()
    answer, backend = generate_answer(query, contexts)
    generation_ms = (time.perf_counter() - start) * 1000
    search.last_query_trace = {
        "retrieval_ms": retrieval_ms,
        "rerank_ms": rerank_ms,
        "generation_ms": generation_ms,
        "generation_backend": backend,
        "candidate_count": len(docs),
        "selected_contexts": selected,
    }
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker, *, resume=False):
    test_set = load_test_set()
    fingerprint = hashlib.sha256(json.dumps({
        "pipeline": search.pipeline_fingerprint, "test_set": test_set,
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    checkpoint_path = Path(".cache/production_queries.json")
    completed = []
    if resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("fingerprint") == fingerprint:
            completed = checkpoint["queries"]
            if len(completed) > len(test_set) or any(
                row["question"] != test_set[i]["question"] or row["ground_truth"] != test_set[i]["ground_truth"]
                for i, row in enumerate(completed)
            ):
                raise ValueError("Query checkpoint does not match the evaluation dataset")
    questions, answers, contexts, truths, traces = [], [], [], [], []
    for i, item in enumerate(test_set, 1):
        if i <= len(completed):
            previous = completed[i - 1]
            answer, evidence, trace = previous["answer"], previous["contexts"], previous["trace"]
            print(f"  [{i}] Reused saved answer: {item['question']}", flush=True)
        else:
            answer, evidence = run_query(item["question"], search, reranker)
            trace = {"question": item["question"], **search.last_query_trace}
            completed.append({
                "question": item["question"], "answer": answer, "contexts": evidence,
                "ground_truth": item["ground_truth"], "trace": trace,
            })
            checkpoint_path.parent.mkdir(exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            temporary.write_text(json.dumps({
                "fingerprint": fingerprint, "queries": completed,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(checkpoint_path)
        questions.append(item["question"])
        answers.append(answer)
        contexts.append(evidence)
        truths.append(item["ground_truth"])
        traces.append(trace)
        print(f"  [{i}] {item['question']}", flush=True)
    start = time.perf_counter()
    results = evaluate_ragas(
        questions, answers, contexts, truths,
        checkpoint_path=".cache/production_metrics.json" if resume else None,
    )
    results["latency"] = {
        "build_ms": search.build_timings,
        "per_query": traces,
        "evaluation_ms": (time.perf_counter() - start) * 1000,
    }
    results["configuration"] = {
        "llm_provider": get_settings().provider,
        "llm_model": get_settings().model,
        "enrichment_backends": search.enrichment_backends,
        "enrichment_cache_reused": search.enrichment_cache_reused,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": reranker.model_name,
        "documents": search.document_count,
        "children": search.chunk_count,
        "parent_context_expansion": True,
        "exclude_superseded_by_default": True,
    }
    for metric in METRICS:
        value = f"{results[metric]:.4f}" if results["metric_sample_counts"][metric] else "N/A (not evaluated)"
        print(f"  {metric}: {value}")
    failures = failure_analysis(results["per_question"], bottom_n=5)
    save_report(results, failures)
    return results


if __name__ == "__main__":
    started = time.perf_counter()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"Total: {time.perf_counter() - started:.1f}s")
