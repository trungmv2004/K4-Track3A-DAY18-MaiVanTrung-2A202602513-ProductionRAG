from __future__ import annotations

"""RAGAS evaluation with explicit availability and evidence-backed diagnostics."""

import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH
from src.llm import create_evaluation_models, get_settings, is_daily_quota_error, safe_error

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class _EvaluationErrors(logging.Handler):
    """Keep errors RAGAS otherwise logs and converts to NaN."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages = []

    def emit(self, record):
        message = safe_error(RuntimeError(record.getMessage()))
        if message not in self.messages:
            self.messages.append(message)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_ragas(
    questions: list[str], answers: list[str], contexts: list[list[str]], ground_truths: list[str],
    *, metric_names: tuple[str, ...] | None = None,
    checkpoint_path: str | None = None,
) -> dict:
    requested = METRICS if metric_names is None else metric_names
    if not requested or set(requested) - set(METRICS):
        raise ValueError("Evaluation metrics must be selected from the four supported RAGAS metrics")
    if len({len(questions), len(answers), len(contexts), len(ground_truths)}) != 1:
        raise ValueError("Evaluation inputs must have the same length")
    rows = [
        EvalResult(q, a, list(c), gt, *([float("nan")] * 4))
        for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
    ]
    unavailable = {
        **dict.fromkeys(METRICS, 0.0),
        "per_question": rows,
        "evaluation_status": "unavailable",
        "metric_sample_counts": dict.fromkeys(METRICS, 0),
    }
    if not questions:
        return {**unavailable, "evaluation_error": "Empty evaluation dataset"}
    settings = get_settings()
    unavailable["evaluation_provider"] = settings.provider
    unavailable["evaluation_model"] = settings.model
    unavailable["evaluation_embedding_model"] = settings.embedding_model
    if not settings.api_key:
        error = (
            "Offline mode: RAGAS requires a local generative LLM server; no cloud requests were made"
            if settings.provider == "offline" else "No API key configured for " + settings.provider
        )
        return {**unavailable, "evaluation_error": error}
    if checkpoint_path is not None:
        if metric_names is not None:
            raise ValueError("Checkpointed evaluation requires all four metrics")
        return _evaluate_checkpointed(questions, answers, contexts, ground_truths, checkpoint_path)
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
        from ragas.run_config import RunConfig

        dataset = Dataset.from_dict(
            {"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths}
        )
        llm, embeddings = create_evaluation_models()
        available_metrics = {metric.name: metric for metric in (
            faithfulness, answer_relevancy, context_precision, context_recall
        )}
        errors = _EvaluationErrors()
        logger = logging.getLogger("ragas.executor")
        logger.addHandler(errors)
        try:
            result = evaluate(
                dataset,
                metrics=[available_metrics[name] for name in requested],
                llm=llm,
                embeddings=embeddings,
                run_config=RunConfig(timeout=300, max_retries=2, max_workers=2),
                raise_exceptions=False,
            )
        finally:
            logger.removeHandler(errors)
        frame = result.to_pandas()
        if len(frame) != len(rows):
            raise ValueError("RAGAS returned an unexpected number of rows")
        for row, record in zip(rows, frame.to_dict(orient="records")):
            for metric in METRICS:
                value = record.get(metric)
                setattr(row, metric, float(value) if value is not None else float("nan"))
        valid = {m: [getattr(row, m) for row in rows if math.isfinite(getattr(row, m))] for m in METRICS}
        status = "complete" if all(len(v) == len(rows) for v in valid.values()) else "partial"
        if not any(valid.values()):
            status = "unavailable"
        return {
            **{m: sum(v) / len(v) if v else 0.0 for m, v in valid.items()},
            "per_question": rows,
            "evaluation_status": status,
            "metric_sample_counts": {m: len(v) for m, v in valid.items()},
            "evaluation_provider": settings.provider,
            "evaluation_model": settings.model,
            "evaluation_embedding_model": settings.embedding_model,
            "evaluated_metrics": list(requested),
            **({"evaluation_error": errors.messages[0], "evaluation_errors": errors.messages}
               if errors.messages else {}),
        }
    except Exception as exc:
        return {**unavailable, "evaluation_error": safe_error(exc)}


def _evaluation_fingerprint(settings, inputs):
    return hashlib.sha256(json.dumps({
        "inputs": inputs, "provider": settings.provider, "model": settings.model,
        "embedding_model": settings.embedding_model,
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _evaluate_checkpointed(questions, answers, contexts, ground_truths, path):
    settings = get_settings()
    inputs = list(zip(questions, answers, contexts, ground_truths))
    fingerprint = _evaluation_fingerprint(settings, inputs)
    checkpoint_path = Path(path)
    stored = {}
    if checkpoint_path.exists():
        candidate = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if candidate.get("fingerprint") == fingerprint:
            stored = candidate
    rows = [EvalResult(q, a, list(c), gt, *([float("nan")] * 4)) for q, a, c, gt in inputs]
    if stored:
        if len(stored["per_question"]) != len(rows):
            raise ValueError("Evaluation checkpoint has an unexpected row count")
        for row, record in zip(rows, stored["per_question"]):
            for metric in METRICS:
                value = record.get(metric)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    setattr(row, metric, value)
    elapsed_ms = stored.get("evaluation_ms", 0)
    reused = sum(all(math.isfinite(getattr(row, m)) for m in METRICS) for row in rows)
    errors = list(stored.get("evaluation_errors", []))
    for i, row in enumerate(rows, 1):
        missing = tuple(m for m in METRICS if not math.isfinite(getattr(row, m)))
        if missing:
            started = time.perf_counter()
            fresh = evaluate_ragas(
                [row.question], [row.answer], [row.contexts], [row.ground_truth], metric_names=missing,
            )
            elapsed_ms += (time.perf_counter() - started) * 1000
            if fresh.get("evaluation_error"):
                errors.append(fresh["evaluation_error"])
            scored = fresh["per_question"][0]
            for metric in missing:
                value = getattr(scored, metric)
                if math.isfinite(value):
                    setattr(row, metric, value)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(_json_safe({
                "fingerprint": fingerprint, "per_question": [asdict(item) for item in rows],
                "evaluation_ms": elapsed_ms,
                "evaluation_errors": errors,
            }), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(checkpoint_path)
            if any(is_daily_quota_error(error) for error in fresh.get("evaluation_errors", []) +
                   [fresh.get("evaluation_error", "")]):
                print("  [RAGAS] Daily quota exhausted; checkpoint saved, stopping evaluation.", flush=True)
                break
        print(f"  [RAGAS {i}/{len(rows)}] " + ("saved" if missing else "reused"), flush=True)
    valid = {m: [getattr(row, m) for row in rows if math.isfinite(getattr(row, m))] for m in METRICS}
    return {
        **{m: sum(values) / len(values) if values else 0.0 for m, values in valid.items()},
        "per_question": rows,
        "evaluation_status": ("complete" if all(len(v) == len(rows) for v in valid.values())
                              else "partial" if any(valid.values()) else "unavailable"),
        "metric_sample_counts": {m: len(v) for m, v in valid.items()},
        "evaluation_provider": settings.provider, "evaluation_model": settings.model,
        "evaluation_embedding_model": settings.embedding_model,
        "evaluation_checkpoint_reused_questions": reused, "evaluation_checkpoint_total_ms": elapsed_ms,
        **({"evaluation_error": errors[-1], "evaluation_errors": errors}
           if errors and not all(len(v) == len(rows) for v in valid.values()) else {}),
    }


def _refresh_report_scores(report):
    rows = report["per_question"]
    valid = {m: [r[m] for r in rows if isinstance(r.get(m), (int, float)) and math.isfinite(r[m])]
             for m in METRICS}
    report["aggregate"] = {m: sum(v) / len(v) if v else None for m, v in valid.items()}
    report["metric_sample_counts"] = {m: len(v) for m, v in valid.items()}
    report["evaluation_status"] = (
        "complete" if rows and all(len(v) == len(rows) for v in valid.values())
        else "partial" if any(valid.values()) else "unavailable"
    )
    report["failures"] = failure_analysis(rows, bottom_n=5)
    if report["evaluation_status"] == "complete":
        report.pop("evaluation_error", None)


def retry_report_missing_metrics(path: str, max_attempts: int = 2, *, checkpoint_path: str | None = None) -> dict:
    """Fill missing cells through real RAGAS calls while preserving every existing score."""
    report_path = Path(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    settings = get_settings()
    if (report.get("evaluation_provider"), report.get("evaluation_model")) != (
        settings.provider, settings.model
    ):
        raise ValueError("Report evaluator does not match current provider/model")
    if report.get("evaluation_embedding_model", settings.embedding_model) != settings.embedding_model:
        raise ValueError("Report evaluator does not match current embedding model")
    _refresh_report_scores(report)
    for attempt in range(1, max_attempts + 1):
        for metric in METRICS:
            indexes = [i for i, row in enumerate(report["per_question"])
                       if not isinstance(row.get(metric), (int, float)) or not math.isfinite(row[metric])]
            if not indexes:
                continue
            rows = [report["per_question"][i] for i in indexes]
            started = time.perf_counter()
            fresh = evaluate_ragas(
                [r["question"] for r in rows], [r["answer"] for r in rows],
                [r["contexts"] for r in rows], [r["ground_truth"] for r in rows], metric_names=(metric,),
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            error = fresh.get("evaluation_error")
            if error:
                report["evaluation_error"] = error
                report.setdefault("evaluation_errors", []).extend(fresh.get("evaluation_errors", [error]))
            updated = []
            for i, scored in zip(indexes, fresh["per_question"]):
                value = getattr(scored, metric)
                if math.isfinite(value):
                    report["per_question"][i][metric] = value
                    updated.append(i + 1)
            report.setdefault("evaluation_retries", []).append({
                "attempt": attempt, "metric": metric, "question_numbers": [i + 1 for i in indexes],
                "filled_question_numbers": updated, "elapsed_ms": elapsed_ms,
                **({"error": error} if error else {}),
            })
            latency = report.setdefault("latency", {})
            latency["evaluation_retry_ms"] = latency.get("evaluation_retry_ms", 0) + elapsed_ms
            _refresh_report_scores(report)
            report_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2,
                                             allow_nan=False), encoding="utf-8")
            print(f"[RAGAS retry] {metric}: filled questions {updated}", flush=True)
            if is_daily_quota_error(error or ""):
                return report
        if report["evaluation_status"] == "complete":
            break
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
        inputs = [(r["question"], r["answer"], r["contexts"], r["ground_truth"])
                  for r in report["per_question"]]
        if checkpoint.get("fingerprint") != _evaluation_fingerprint(settings, inputs):
            raise ValueError("Metric checkpoint does not match report data/evaluator")
        checkpoint["per_question"] = report["per_question"]
        retries = report.get("evaluation_retries", [])
        applied = checkpoint.get("applied_report_retry_count", 0)
        checkpoint["evaluation_ms"] += sum(item["elapsed_ms"] for item in retries[applied:])
        checkpoint["applied_report_retry_count"] = len(retries)
        Path(checkpoint_path).write_text(json.dumps(_json_safe(checkpoint), ensure_ascii=False,
                                                   indent=2, allow_nan=False), encoding="utf-8")
        report["evaluation_checkpoint_total_ms"] = checkpoint["evaluation_ms"]
        report_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2,
                                         allow_nan=False), encoding="utf-8")
    report_path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2,
                                     allow_nan=False), encoding="utf-8")
    return report


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    tree = {
        "faithfulness": (
            "LLM hallucinating",
            "Tighten grounded prompt and lower temperature",
            "Answer → supported by context? → generation",
        ),
        "context_recall": (
            "Missing relevant chunks",
            "Improve chunking, hybrid retrieval and multi-hop coverage",
            "Answer → context complete? → retrieval/chunking",
        ),
        "context_precision": (
            "Too many irrelevant chunks",
            "Add reranking or version metadata filters",
            "Answer → context relevant? → ranking/filtering",
        ),
        "answer_relevancy": (
            "Answer does not match question",
            "Improve prompt template and answer synthesis",
            "Answer → addresses query? → generation/query intent",
        ),
    }
    failures = []
    for item in eval_results:
        row = asdict(item) if isinstance(item, EvalResult) else item
        values = {m: row.get(m) for m in METRICS}
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values.values()):
            continue
        worst = min(values, key=values.get)
        diagnosis, fix, branch = tree[worst]
        failures.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "ground_truth": row["ground_truth"],
                "contexts": row["contexts"],
                "worst_metric": worst,
                "score": values[worst],
                "average_score": sum(values.values()) / 4,
                "diagnosis": diagnosis,
                "suggested_fix": fix,
                "error_tree": branch,
            }
        )
    return sorted(failures, key=lambda row: row["average_score"])[: max(0, bottom_n)]


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    report_path = Path(path)
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        previous_rows = previous.get("per_question", [])
        if previous_rows and previous.get("evaluation_status") == "complete" and all(
            isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric])
            for row in previous_rows for metric in METRICS
        ):
            archive = report_path.with_name("last_complete_" + report_path.name)
            archive.write_text(json.dumps(previous, ensure_ascii=False, indent=2, allow_nan=False),
                               encoding="utf-8")
    counts = results.get("metric_sample_counts", {})
    aggregate = {m: results.get(m) if counts.get(m, 1) else None for m in METRICS}
    report = {
        "aggregate": aggregate,
        "num_questions": len(results.get("per_question", [])),
        "per_question": [
            asdict(row) if isinstance(row, EvalResult) else row for row in results.get("per_question", [])
        ],
        "failures": failures,
        **{k: v for k, v in results.items() if k not in METRICS and k != "per_question"},
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    print(f"Loaded {len(load_test_set())} test questions")
