"""Regression checks for hierarchy, evaluation availability and model integration."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from qdrant_client import QdrantClient

import config
from src import m4_eval, m5_enrichment, pipeline
from src.m1_chunking import Chunk, chunk_hierarchical, chunk_structure_aware
from src.m2_search import BM25Search, DenseSearch, SearchResult, reciprocal_rank_fusion
from src.m3_rerank import CrossEncoderReranker


def test_hierarchy_bounds_and_cross_document_ids():
    text = "Một đoạn văn dài không có xuống dòng. " * 100
    a, children = chunk_hierarchical(text, parent_size=120, child_size=40, metadata={"source": "a.md"})
    b, _ = chunk_hierarchical(text, parent_size=120, child_size=40, metadata={"source": "b.md"})
    assert all(0 < len(p.text) <= 120 for p in a)
    assert all(0 < len(c.text) <= 40 for c in children)
    assert {p.metadata["parent_id"] for p in a}.isdisjoint(p.metadata["parent_id"] for p in b)
    assert "".join(c.text.replace(" ", "") for c in children) == text.replace(" ", "")


def test_structure_keeps_fenced_headers_and_tables():
    text = "# Real\n```markdown\n## Inside code\n```\n| A | B |\n|---|---|\n|1|2|\n## Next\nBody"
    chunks = chunk_structure_aware(text)
    assert len(chunks) == 2
    assert "## Inside code" in chunks[0].text and "|1|2|" in chunks[0].text


def test_empty_and_invalid_sizes():
    assert chunk_hierarchical("") == ([], [])
    with pytest.raises(ValueError):
        chunk_hierarchical("x", parent_size=10, child_size=10)
    assert reciprocal_rank_fusion([[SearchResult("x", 1, {}, "dense")]], top_k=0) == []


def test_bm25_small_corpus_and_case():
    search = BM25Search()
    search.index([{"text": "nghỉ phép", "metadata": {}}])
    assert search.search("NGHỈ PHÉP")[0].score > 0
    search.index([])
    assert search.search("nghỉ phép") == []


def test_rrf_scores_and_same_text_different_sources():
    a = SearchResult("same", 0.3, {"source": "a"}, "bm25")
    b = SearchResult("same", 0.9, {"source": "b"}, "dense")
    result = reciprocal_rank_fusion([[a, a], [a, b]])
    assert len(result) == 2
    assert result[0].score == pytest.approx(2 / 61)


def test_dense_real_qdrant_index_query_and_reindex():
    encoder = SimpleNamespace(
        get_sentence_embedding_dimension=lambda: 2,
        encode=lambda texts, **kwargs: (
            np.array([1.0, 0.0])
            if isinstance(texts, str)
            else np.array([[1.0, 0.0], [0.0, 1.0]][: len(texts)])
        ),
    )
    search = DenseSearch.__new__(DenseSearch)
    search.client, search._encoder = QdrantClient(":memory:"), encoder
    try:
        search.index([{"text": "leave", "metadata": {"source": "a"}}, {"text": "VPN"}], "test")
        found = search.search("leave", collection="test")
        assert found[0].text == "leave" and found[0].metadata == {"source": "a"}
        assert found[0].method == "dense"
        search.index([], "test")
        assert search.search("leave", collection="test") == []
    finally:
        search.client.close()


def test_reranker_scalar_and_metadata():
    reranker = CrossEncoderReranker()
    reranker._model = SimpleNamespace(predict=lambda pairs: 0.7)
    result = reranker.rerank("q", [{"text": "a", "score": 0.2, "metadata": {"source": "x"}}])
    assert result[0].rerank_score == pytest.approx(0.7)
    assert result[0].original_score == 0.2 and result[0].rank == 1
    assert reranker.rerank("q", [], top_k=3) == []


def test_missing_key_report_preserves_rows_without_fake_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    results = m4_eval.evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    assert results["evaluation_status"] == "unavailable"
    assert m4_eval.failure_analysis(results["per_question"]) == []
    path = tmp_path / "report.json"
    m4_eval.save_report(results, [], str(path))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["num_questions"] == 1 and report["per_question"][0]["contexts"] == ["c"]
    assert all(value is None for value in report["aggregate"].values())
    with pytest.raises(ValueError):
        m4_eval.evaluate_ragas(["q"], [], [], [])


def test_combined_single_call_preserves_authoritative_metadata(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    call = Mock(
        return_value=json.dumps(
            {
                "summary": "S",
                "questions": ["Q?"],
                "context": "C",
                "metadata": {"source": "wrong", "parent_id": "wrong"},
            }
        )
    )
    monkeypatch.setattr(m5_enrichment, "_llm", call)
    enriched = m5_enrichment.enrich_chunks(
        [{"text": "Original", "metadata": {"source": "a.md", "parent_id": "p"}}]
    )[0]
    assert call.call_count == 1
    assert enriched.auto_metadata["source"] == "a.md" and enriched.auto_metadata["parent_id"] == "p"
    assert "Original" in enriched.enriched_text and "Q?" in enriched.enriched_text
    call.side_effect = ValueError("invalid JSON")
    assert m5_enrichment.enrich_chunks([{"text": "Original"}])[0].original_text == "Original"


def test_pipeline_restores_distinct_parent_contexts(monkeypatch):
    from src.m3_rerank import RerankResult

    search = SimpleNamespace(
        search=lambda query: [
            SearchResult("a1", 1, {"parent_id": "a"}, "hybrid"),
            SearchResult("a2", 0.9, {"parent_id": "a"}, "hybrid"),
            SearchResult("b1", 0.8, {"parent_id": "b"}, "hybrid"),
        ],
        parents_by_id={"a": Chunk("Full A"), "b": Chunk("Full B")},
    )
    reranker = SimpleNamespace(
        rerank=lambda query, docs, top_k: [
            RerankResult(d["text"], d["score"], 1.0, d["metadata"], i + 1) for i, d in enumerate(docs)
        ]
    )
    monkeypatch.setattr(pipeline, "generate_answer", lambda q, c: (c[0], "test"))
    _, contexts = pipeline.run_query("q", search, reranker)
    assert contexts == ["Full A", "Full B"]


def test_policy_versions_inferred_from_document_metadata():
    documents = [
        {
            "text": "# Chính sách A (2023)\nPhiên bản: 1.0 | Ngày hiệu lực: 01/01/2023",
            "metadata": {"source": "a1"},
        },
        {
            "text": "# Chính sách A (2024)\nPhiên bản: 2.0 | Ngày hiệu lực: 01/01/2024",
            "metadata": {"source": "a2"},
        },
        {"text": "# Chính sách B\nNội dung khác", "metadata": {"source": "b"}},
    ]
    pipeline._document_metadata(documents)
    assert documents[0]["metadata"]["superseded"] is True
    assert documents[1]["metadata"]["superseded"] is False
    assert documents[2]["metadata"]["superseded"] is False


def test_ragas_contract_and_partial_metrics(monkeypatch):
    import pandas as pd
    import ragas
    from ragas.validation import validate_column_dtypes

    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    calls = []

    def fake_evaluate(dataset, **kwargs):
        validate_column_dtypes(dataset)
        calls.append(kwargs)
        frame = dataset.to_pandas()
        frame["faithfulness"] = [0.8, 0.6]
        frame["answer_relevancy"] = [0.7, float("nan")]
        frame["context_precision"] = [0.6, 0.4]
        frame["context_recall"] = [0.5, 0.3]
        assert isinstance(frame, pd.DataFrame)
        return SimpleNamespace(to_pandas=lambda: frame)

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    result = m4_eval.evaluate_ragas(["q1", "q2"], ["a1", "a2"], [["c1"], ["c2"]], ["gt1", "gt2"])
    assert {m.name for m in calls[0]["metrics"]} == set(m4_eval.METRICS)
    assert result["evaluation_status"] == "partial"
    assert result["faithfulness"] == pytest.approx(0.7)
    assert result["answer_relevancy"] == pytest.approx(0.7)
    assert result["metric_sample_counts"]["answer_relevancy"] == 1
    failures = m4_eval.failure_analysis(result["per_question"])
    assert len(failures) == 1 and failures[0]["question"] == "q1"


def test_enrichment_cache_requires_exact_source_content_and_metadata():
    chunks = [{"text": "Original", "metadata": {"source": "a.md", "parent_id": "p", "version": "2"}}]
    cached = [{"text": "Enriched Original", "metadata": {
        **chunks[0]["metadata"], "original_text": "Original", "enrichment_backend": "gemini",
    }}]
    assert pipeline._validate_enrichment_cache(chunks, cached) == cached
    wrong_text = [{"text": "Enriched", "metadata": {**cached[0]["metadata"], "original_text": "changed"}}]
    with pytest.raises(ValueError, match="source chunks"):
        pipeline._validate_enrichment_cache(chunks, wrong_text)
    wrong_version = [{"text": "Enriched", "metadata": {**cached[0]["metadata"], "version": "1"}}]
    with pytest.raises(ValueError, match="stale source metadata"):
        pipeline._validate_enrichment_cache(chunks, wrong_version)


def test_selected_ragas_metric_does_not_request_other_metrics(monkeypatch):
    import ragas

    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")

    def fake_evaluate(dataset, **kwargs):
        assert [m.name for m in kwargs["metrics"]] == ["faithfulness"]
        frame = dataset.to_pandas()
        frame["faithfulness"] = [0.8]
        return SimpleNamespace(to_pandas=lambda: frame)

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    result = m4_eval.evaluate_ragas(["q"], ["a"], [["c"]], ["gt"], metric_names=("faithfulness",))
    assert result["metric_sample_counts"] == {
        "faithfulness": 1, "answer_relevancy": 0, "context_precision": 0, "context_recall": 0,
    }
    with pytest.raises(ValueError, match="supported RAGAS metrics"):
        m4_eval.evaluate_ragas(["q"], ["a"], [["c"]], ["gt"], metric_names=("unknown",))


def test_retry_report_fills_missing_cells_without_replacing_valid_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    path = tmp_path / "report.json"
    report = {
        "evaluation_provider": "openai", "evaluation_model": "gpt-4o-mini",
        "evaluation_status": "partial", "num_questions": 1,
        "per_question": [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "gt",
                          "faithfulness": None, "answer_relevancy": .8, "context_precision": .7,
                          "context_recall": .6}],
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    checkpoint_path = tmp_path / "metrics.json"
    inputs = [("q", "a", ["c"], "gt")]
    checkpoint_path.write_text(json.dumps({
        "fingerprint": m4_eval._evaluation_fingerprint(m4_eval.get_settings(), inputs),
        "per_question": report["per_question"], "evaluation_ms": 5,
    }), encoding="utf-8")

    def score(q, a, c, gt, **kwargs):
        assert kwargs["metric_names"] == ("faithfulness",)
        return {"per_question": [m4_eval.EvalResult(q[0], a[0], c[0], gt[0], .9, 0, 0, 0)]}

    monkeypatch.setattr(m4_eval, "evaluate_ragas", score)
    updated = m4_eval.retry_report_missing_metrics(str(path), checkpoint_path=str(checkpoint_path))
    assert updated["evaluation_status"] == "complete"
    assert updated["aggregate"] == {
        "faithfulness": .9, "answer_relevancy": .8, "context_precision": .7, "context_recall": .6,
    }
    assert updated["per_question"][0]["answer"] == "a"
    assert updated["evaluation_retries"][0]["filled_question_numbers"] == [1]
    stored = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert stored["per_question"][0]["faithfulness"] == .9
    assert stored["per_question"][0]["answer_relevancy"] == .8
    m4_eval.retry_report_missing_metrics(str(path), checkpoint_path=str(checkpoint_path))
    assert json.loads(checkpoint_path.read_text(encoding="utf-8"))["evaluation_ms"] == stored["evaluation_ms"]


def test_evaluation_checkpoint_resumes_only_unmeasured_cells(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "test-key")
    calls = []

    def score(q, a, c, gt, **kwargs):
        calls.append((q[0], kwargs["metric_names"]))
        faithfulness = float("nan") if q[0] == "q2" and len(calls) == 2 else .9
        return {"per_question": [m4_eval.EvalResult(q[0], a[0], c[0], gt[0], faithfulness, .8, .7, .6)]}

    monkeypatch.setattr(m4_eval, "evaluate_ragas", score)
    path = str(tmp_path / "checkpoint.json")
    args = (["q1", "q2"], ["a1", "a2"], [["c1"], ["c2"]], ["gt1", "gt2"], path)
    first = m4_eval._evaluate_checkpointed(*args)
    assert first["evaluation_status"] == "partial"
    second = m4_eval._evaluate_checkpointed(*args)
    assert second["evaluation_status"] == "complete"
    assert second["evaluation_checkpoint_reused_questions"] == 1
    assert calls[-1] == ("q2", ("faithfulness",))
    assert len(calls) == 3


def test_query_checkpoint_avoids_regeneration_and_invalidates_changed_inputs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    test_set = [{"question": "q1", "ground_truth": "gt1"}, {"question": "q2", "ground_truth": "gt2"}]
    monkeypatch.setattr(pipeline, "load_test_set", lambda: test_set)
    calls = []
    search = SimpleNamespace(
        pipeline_fingerprint="current-data-and-model", build_timings={},
        enrichment_backends={"test": 2}, enrichment_cache_reused=True, document_count=1, chunk_count=2,
    )

    def query(question, search, reranker):
        calls.append(question)
        search.last_query_trace = {"generation_backend": "test", "selected_contexts": []}
        return "answer " + question, ["context"]

    def score(questions, answers, contexts, truths, **kwargs):
        return {
            **{m: .8 for m in m4_eval.METRICS}, "evaluation_status": "complete",
            "metric_sample_counts": {m: len(questions) for m in m4_eval.METRICS},
            "per_question": [m4_eval.EvalResult(q, a, c, gt, .8, .8, .8, .8)
                             for q, a, c, gt in zip(questions, answers, contexts, truths)],
        }

    monkeypatch.setattr(pipeline, "run_query", query)
    monkeypatch.setattr(pipeline, "evaluate_ragas", score)
    reranker = SimpleNamespace(model_name="test")
    pipeline.evaluate_pipeline(search, reranker, resume=True)
    pipeline.evaluate_pipeline(search, reranker, resume=True)
    assert calls == ["q1", "q2"]
    test_set[1]["ground_truth"] = "changed ground truth"
    pipeline.evaluate_pipeline(search, reranker, resume=True)
    assert calls == ["q1", "q2", "q1", "q2"]
