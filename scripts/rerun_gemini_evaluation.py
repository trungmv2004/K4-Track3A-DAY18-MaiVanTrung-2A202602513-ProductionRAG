"""Re-evaluate saved baseline answers and regenerate production using cached enrichment."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.llm import check_evaluation_provider, get_settings
from src.m4_eval import evaluate_ragas, failure_analysis, retry_report_missing_metrics, save_report
from src.pipeline import build_pipeline, evaluate_pipeline


def main():
    settings = get_settings()
    if settings.provider != "gemini":
        raise ValueError("This runner requires the Gemini provider")
    check_evaluation_provider()
    previous = json.loads(Path("reports/naive_baseline_report.json").read_text(encoding="utf-8"))
    rows = previous["per_question"]
    print(f"[Baseline] Re-evaluating {len(rows)} saved answers with {settings.model}", flush=True)
    started = time.perf_counter()
    checkpoint = ".cache/baseline_metrics.json"
    scores = evaluate_ragas(
        [r["question"] for r in rows], [r["answer"] for r in rows],
        [r["contexts"] for r in rows], [r["ground_truth"] for r in rows],
        checkpoint_path=checkpoint,
    )
    scores["configuration"] = previous["configuration"]
    scores["latency"] = {
        **previous["latency"], "evaluation_ms": (time.perf_counter() - started) * 1000,
        "evaluation_retry_ms": 0,
    }
    scores["evaluation_provenance"] = {
        "note": "Saved baseline answers re-evaluated; generation was not repeated",
        "generation_model": previous["configuration"]["llm_model"],
        "previous_evaluation_model": previous["evaluation_model"],
    }
    save_report(scores, failure_analysis(scores["per_question"], 5), "reports/naive_baseline_report.json")
    if scores["evaluation_status"] != "complete":
        fixed = retry_report_missing_metrics("reports/naive_baseline_report.json", checkpoint_path=checkpoint)
        if fixed["evaluation_status"] != "complete":
            return 1
    print("[Baseline] All 80 metric cells complete", flush=True)

    cached = json.loads(Path(".cache/gemini_enriched_index.json").read_text(encoding="utf-8"))
    search, reranker = build_pipeline(indexed_cache=cached)
    production = evaluate_pipeline(search, reranker, resume=True)
    report_path = Path("reports/ragas_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["configuration"]["enrichment_model"] = "gemini-3.5-flash-lite"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if production["evaluation_status"] != "complete":
        report = retry_report_missing_metrics(str(report_path), checkpoint_path=".cache/production_metrics.json")
    print("[Production] " + report["evaluation_status"], flush=True)
    return 0 if report["evaluation_status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
