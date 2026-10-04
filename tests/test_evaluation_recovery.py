import json
import logging
from types import SimpleNamespace

import pytest

import config
from src import llm, m4_eval


def test_ragas_nan_retains_redacted_executor_error(monkeypatch):
    import ragas

    monkeypatch.setattr(config, "OPENAI_API_KEY", "secret-test-key")
    logger = logging.getLogger("ragas.executor")
    before = list(logger.handlers)

    def fail(dataset, **kwargs):
        logger.error("Job failed: GenerateRequestsPerDay secret-test-key")
        frame = dataset.to_pandas()
        return SimpleNamespace(to_pandas=lambda: frame)

    monkeypatch.setattr(ragas, "evaluate", fail)
    result = m4_eval.evaluate_ragas(["q"], ["a"], [["c"]], ["gt"])
    assert result["evaluation_status"] == "unavailable"
    assert "GenerateRequestsPerDay" in result["evaluation_error"]
    assert "secret-test-key" not in result["evaluation_error"]
    assert "[REDACTED]" in result["evaluation_error"]
    assert logger.handlers == before


def test_checkpoint_stops_after_daily_quota_and_preserves_unscored_rows(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "fake-key")
    calls = []

    def fail(q, a, c, gt, **kwargs):
        calls.append(q)
        return {
            "per_question": [m4_eval.EvalResult(q[0], a[0], c[0], gt[0], *([float("nan")] * 4))],
            "evaluation_error": "GenerateRequestsPerDay quota exhausted",
        }

    monkeypatch.setattr(m4_eval, "evaluate_ragas", fail)
    path = tmp_path / "checkpoint.json"
    result = m4_eval._evaluate_checkpointed(
        ["q1", "q2"], ["a1", "a2"], [["c1"], ["c2"]], ["gt1", "gt2"], str(path),
    )
    assert calls == [["q1"]]
    assert result["evaluation_status"] == "unavailable"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert len(stored["per_question"]) == 2
    assert all(row[m] is None for row in stored["per_question"] for m in m4_eval.METRICS)
    assert "GenerateRequestsPerDay" in stored["evaluation_errors"][0]


def test_retry_stops_on_daily_quota_without_replacing_existing_score(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OPENAI_API_KEY", "fake-key")
    path = tmp_path / "report.json"
    path.write_text(json.dumps({
        "evaluation_provider": "openai", "evaluation_model": "gpt-4o-mini",
        "evaluation_status": "partial",
        "per_question": [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "gt",
                          "faithfulness": None, "answer_relevancy": .9,
                          "context_precision": None, "context_recall": None}],
    }), encoding="utf-8")
    calls = []

    def fail(q, a, c, gt, **kwargs):
        calls.append(kwargs["metric_names"])
        return {
            "per_question": [m4_eval.EvalResult(q[0], a[0], c[0], gt[0], *([float("nan")] * 4))],
            "evaluation_error": "GenerateRequestsPerDay quota exhausted",
        }

    monkeypatch.setattr(m4_eval, "evaluate_ragas", fail)
    result = m4_eval.retry_report_missing_metrics(str(path))
    assert calls == [("faithfulness",)]
    assert result["per_question"][0]["answer_relevancy"] == .9
    assert result["aggregate"]["faithfulness"] is None
    assert "GenerateRequestsPerDay" in result["evaluation_error"]


def test_preflight_does_not_request_embeddings_after_chat_quota_error(monkeypatch):
    class Client:
        def __enter__(self):
            def fail(**kwargs):
                raise RuntimeError("GenerateRequestsPerDay")
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=fail))
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(llm, "create_client", lambda **kwargs: Client())

    def forbidden():
        raise AssertionError("Embeddings should not run after a failed chat probe")

    monkeypatch.setattr(llm, "create_evaluation_models", forbidden)
    with pytest.raises(RuntimeError, match="GenerateRequestsPerDay"):
        llm.check_evaluation_provider()
    assert not llm.is_daily_quota_error("GenerateRequestsPerMinute")


def test_main_failed_preflight_keeps_reports_byte_identical(monkeypatch, tmp_path):
    import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "fake-key")
    folder = tmp_path / "reports"
    folder.mkdir()
    path = folder / "ragas_report.json"
    contents = b'{"evaluation_status": "complete", "evidence": "previous run"}'
    path.write_bytes(contents)

    def fail():
        raise RuntimeError("GenerateRequestsPerDay")

    monkeypatch.setattr(llm, "check_evaluation_provider", fail)
    assert main.main() == 1
    assert path.read_bytes() == contents


def test_eval_only_complete_reports_do_not_use_api(monkeypatch, tmp_path):
    import main

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "fake-key")
    folder = tmp_path / "reports"
    folder.mkdir()
    row = {"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "gt",
           **dict.fromkeys(m4_eval.METRICS, .8)}
    report = {"evaluation_provider": "openai", "evaluation_model": "gpt-4o-mini",
              "evaluation_status": "complete", "metric_sample_counts": dict.fromkeys(m4_eval.METRICS, 1),
              "per_question": [row]}
    for name in ("ragas_report.json", "naive_baseline_report.json"):
        (folder / name).write_text(json.dumps(report), encoding="utf-8")

    def forbidden():
        raise AssertionError("Complete reports must not consume API quota")

    monkeypatch.setattr(llm, "check_evaluation_provider", forbidden)
    monkeypatch.setattr(llm, "create_client", forbidden)
    assert main.main(eval_only=True) == 0


def test_save_failed_run_archives_complete_report_with_original_inputs(tmp_path):
    path = tmp_path / "ragas_report.json"
    old = {"evaluation_status": "complete", "per_question": [
        {"question": "previous q", "answer": "previous a", "contexts": ["previous c"],
         "ground_truth": "previous gt", **dict.fromkeys(m4_eval.METRICS, .9)},
    ]}
    path.write_text(json.dumps(old), encoding="utf-8")
    unavailable = {
        "evaluation_status": "unavailable", "metric_sample_counts": dict.fromkeys(m4_eval.METRICS, 0),
        "per_question": [m4_eval.EvalResult("new q", "new a", ["new c"], "new gt", *([float("nan")] * 4))],
    }
    m4_eval.save_report(unavailable, [], str(path))
    assert json.loads((tmp_path / "last_complete_ragas_report.json").read_text(encoding="utf-8")) == old
    current = json.loads(path.read_text(encoding="utf-8"))
    assert current["per_question"][0]["question"] == "new q"
    assert all(value is None for value in current["aggregate"].values())
