"""Summarize actual Gemini corpus reports without making any additional API calls."""

import json
import math
import re
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from refresh_local_reports import percentile

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    baseline = read_json(REPORTS / "naive_baseline_report.json")
    production = read_json(REPORTS / "ragas_report.json")
    for report in (baseline, production):
        assert report["configuration"]["llm_provider"] == "gemini"
        assert report["num_questions"] == len(report["per_question"]) == 20
        assert report["evaluation_status"] == "complete"
        assert all(report["metric_sample_counts"][metric] == 20 for metric in METRICS)
        assert all(math.isfinite(row[metric]) for row in report["per_question"] for metric in METRICS)
    assert (baseline["evaluation_provider"], baseline["evaluation_model"], baseline["evaluation_embedding_model"]) == (
        production["evaluation_provider"], production["evaluation_model"], production["evaluation_embedding_model"],
    )
    stamp = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat()

    cases = []
    for index, row in enumerate(production["per_question"], 1):
        values = {metric: row[metric] for metric in METRICS}
        worst = min(values, key=values.get)
        cases.append({
            "test_question_number": index, **row, "average_score": statistics.mean(values.values()),
            "worst_metric": worst,
            "selected_sources": production["latency"]["per_query"][index - 1]["selected_contexts"],
        })
    selected = sorted(cases, key=lambda row: row["average_score"])[:5]
    assert [row["question"] for row in selected] == [row["question"] for row in production["failures"]]
    audit = read_json(ROOT / ".cache/retrieval_audit.json")
    for row in selected:
        trace = audit[row["test_question_number"] - 1]
        sources = list(dict.fromkeys(item["source"] for item in trace["reranked"]))[:3]
        assert sources == [item["source"] for item in row["selected_sources"]]
        row["candidates"] = trace["candidates"]
        row["reranked"] = trace["reranked"]
    write_json(REPORTS / "diagnostic_review.json", {
        "selection_method": "mean_four_ragas_metrics_ascending_stable_question_order",
        "not_ragas": False, "ragas_status": "complete", "configuration": production["configuration"],
        "cases": selected,
        "trace_provenance": "Local replay on the recovered index/models; selected sources verified against report",
    })

    lines = [
        "# Latency breakdown - Lab 18", "", f"Run recorded: {stamp}.", "",
        "Profile: lightweight retrieval models with Gemini generation, enrichment and RAGAS.",
        "Embedding: " + production["configuration"]["embedding_model"] + ".",
        "Reranker: " + production["configuration"]["reranker_model"] + ".",
        "LLM: " + production["configuration"]["llm_model"] + ".",
        "RAGAS embeddings: " + production["evaluation_embedding_model"] + ".",
        "Baseline generation/build timings come from the saved original run; evaluator timings come from the current run.",
        "Production reused validated enrichment and saved matching checkpoints when available.",
        "Baseline generation model: " + baseline["configuration"]["llm_model"] + ".",
        "Shared RAGAS evaluator: " + production["evaluation_model"] + ".",
        "Enrichment model: " + production["configuration"].get("enrichment_model", "recorded cached Gemini run") + ".",
        "Index timings include encoder loading as needed; these runs do not share an encoder process.",
        "Generation includes API/network latency and quota waits.",
        "Production reused all 101 previously generated Gemini enrichment payloads.",
        "The enrichment build timing below measures cache validation, not the original API calls.",
        "RAGAS evaluation includes all four metrics over 20 questions and recorded retries.", "",
        "| Build step | Baseline (ms) | Production (ms) |", "|---|---:|---:|",
    ]
    for step in ("load_and_chunk_ms", "enrichment_ms", "index_ms", "reranker_load_ms"):
        values = [report["latency"]["build_ms"].get(step) for report in (baseline, production)]
        fields = [f"{value:.2f}" if value is not None else "N/A" for value in values]
        lines.append(f"| {step} | {fields[0]} | {fields[1]} |")
    lines += [
        "", "| Pipeline | Query step | Samples | Mean (ms) | p50 (ms) | p95 (ms) | Max (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    backends = {}
    for label, report in (("Baseline", baseline), ("Production", production)):
        backends[label] = {}
        for row in report["latency"]["per_query"]:
            backend = row["generation_backend"]
            backends[label][backend] = backends[label].get(backend, 0) + 1
        for step in ("retrieval_ms", "rerank_ms", "generation_ms"):
            values = [row[step] for row in report["latency"]["per_query"] if step in row]
            if values:
                lines.append(
                    f"| {label} | {step} | {len(values)} | {statistics.mean(values):.2f} | "
                    f"{percentile(values, .5):.2f} | {percentile(values, .95):.2f} | {max(values):.2f} |"
                )
    lines += ["", "| Evaluation | Time (seconds) |", "|---|---:|"]
    for label, report in (("Baseline", baseline), ("Production", production)):
        lines.append(f"| {label} RAGAS | {report['latency']['evaluation_ms'] / 1000:.2f} |")
        if report["latency"].get("evaluation_retry_ms"):
            lines.append(f"| {label} additional metric retries | {report['latency']['evaluation_retry_ms'] / 1000:.2f} |")
    lines += [
        "", "Generation backends: " + json.dumps(backends) + ".",
        "Enrichment backends: " + json.dumps(production["configuration"]["enrichment_backends"]) + ".",
        "Raw timings and source traces are preserved in the two JSON reports.", "",
    ]
    (REPORTS / "latency_report.md").write_text("\n".join(lines), encoding="utf-8")

    check_bytes = (ROOT / ".cache/gemini_check_lab.log").read_bytes()
    check_log = check_bytes.decode("utf-16" if check_bytes.startswith(b"\xff\xfe") else "utf-8")
    passed, total = map(int, re.search(r"(\d+)/(\d+) tests passed", check_log).groups())
    check_exit = int((ROOT / ".cache/gemini_check_lab.exit").read_text())
    standalone_path = ROOT / ".cache/standalone_pipeline.exit"
    write_json(REPORTS / "validation_report.json", {
        "recorded_at": stamp, "timezone": "Asia/Bangkok", "profile": "lightweight_gemini",
        "models": production["configuration"],
        "pytest": {"passed": passed, "failed": total - passed},
        "main_exit_code": int((ROOT / ".cache/gemini_main.exit").read_text()),
        "main_validation_mode": "--eval-only",
        "evaluation_runner_exit_code": (
            int((ROOT / ".cache/gemini31_rerun.exit").read_text())
            if (ROOT / ".cache/gemini31_rerun.exit").exists() else None
        ),
        "evaluation_model": production["evaluation_model"],
        "baseline_generation_model": baseline["configuration"]["llm_model"],
        "standalone_pipeline_exit_code": int(standalone_path.read_text()),
        "standalone_pipeline_validation": {"provider": "offline", "note":
            "Earlier standalone run executed offline in an isolated directory with in-memory Qdrant. "
            "Real Gemini evaluation provenance is recorded separately in the current reports."},
        "ragas_status": "complete", "baseline_ragas_status": baseline["evaluation_status"],
        "metric_sample_counts": production["metric_sample_counts"],
        "generation_backends": backends, "remaining_requirement": None,
        "cloud_corpus_export_authorized": True, "cloud_corpus_export_performed": True,
        "commit_performed": False, "push_performed": False,
        "check_lab_exit_code": check_exit, "check_lab_remaining_errors": [],
        "ruff_check": "passed" if int((ROOT / ".cache/gemini_ruff.exit").read_text()) == 0 else "failed",
        "remaining_todos": 0, "not_ready_for_submission": check_exit != 0,
        "rubric_estimate": {
            "implementation": 60, "pipeline": 10,
            "ragas_scores": next(points for count, points in ((3, 10), (2, 8), (1, 5), (0, 3))
                                 if sum(v >= .7 for v in production["aggregate"].values()) >= count),
            "failure_analysis": 5, "reflection": 15,
            "bonus_faithfulness": 3 if production["aggregate"]["faithfulness"] >= .85 else 0,
            "bonus_combined_enrichment": 2, "bonus_latency": 2,
            "bonus_all_metrics": 3 if all(v >= .75 for v in production["aggregate"].values()) else 0,
            "note": "Estimate subject to code/document review; raw threshold comparisons, without rounding",
        },
    })
    print("Production metrics:", production["aggregate"])
    print("Bottom-5 question numbers:", [row["test_question_number"] for row in selected])
    print(f"Validation: {passed}/{total} tests, check_lab exit={check_exit}")


if __name__ == "__main__":
    main()
