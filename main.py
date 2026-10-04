"""
Lab 18: Production RAG Pipeline — Main Entry Point
===================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py
"""

import json
import math
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def main(resume=False, eval_only=False):
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()

    os.makedirs("reports", exist_ok=True)
    from src.llm import check_evaluation_provider, get_settings, safe_error

    paths = ("reports/naive_baseline_report.json", "reports/ragas_report.json")
    needs_evaluation = False
    if eval_only:
        from src.m4_eval import METRICS

        for path in paths:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
            needs_evaluation |= report.get("evaluation_status") != "complete" or any(
                report.get("metric_sample_counts", {}).get(metric) != len(report["per_question"])
                for metric in METRICS
            )
            needs_evaluation |= any(
                not isinstance(row.get(metric), (int, float)) or not math.isfinite(row[metric])
                for row in report["per_question"] for metric in METRICS
            )
    if needs_evaluation or (not eval_only and not resume and get_settings().provider != "offline"):
        try:
            check_evaluation_provider()
        except Exception as exc:
            print("\nRAGAS provider check failed: " + safe_error(exc))
            print("Báo cáo hiện có được giữ nguyên. Sửa lỗi provider/quota trước khi chạy lại.")
            return 1
    if eval_only:
        from src.m4_eval import retry_report_missing_metrics

        reports = [retry_report_missing_metrics(path) for path in paths]
        for path, report in zip(paths, reports):
            print(f"{path}: {report['evaluation_status']} — {report['metric_sample_counts']}")
        return 0 if all(report["evaluation_status"] == "complete" for report in reports) else 1

    # Step 1: Basic Baseline
    print("\n📌 STEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    from naive_baseline import main as run_baseline

    if resume and os.path.exists("reports/naive_baseline_report.json"):
        from src.m4_eval import retry_report_missing_metrics

        retry_report_missing_metrics("reports/naive_baseline_report.json")
    else:
        run_baseline()

    # Step 2: Production Pipeline
    print("\n📌 STEP 2: Running Production Pipeline...")
    print("-" * 40)
    from src.pipeline import build_pipeline, evaluate_pipeline

    cache_path = ".cache/production_enriched_index.json"
    if not os.path.exists(cache_path):
        cache_path = ".cache/gemini_enriched_index.json"
    indexed_cache = None
    if resume and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            indexed_cache = json.load(handle)
    search, reranker = build_pipeline(indexed_cache=indexed_cache)
    prod_results = evaluate_pipeline(search, reranker, resume=resume)
    if resume and prod_results.get("evaluation_status") != "complete":
        from src.m4_eval import retry_report_missing_metrics

        prod_results = retry_report_missing_metrics(
            "reports/ragas_report.json", checkpoint_path=".cache/production_metrics.json"
        )

    # Ensure reports are located in reports/
    for f in ["ragas_report.json", "naive_baseline_report.json"]:
        if os.path.exists(f):
            os.replace(f, f"reports/{f}")

    # Step 3: Comparison
    print("\n📌 STEP 3: Comparison")
    print("-" * 40)
    naive_path = "reports/naive_baseline_report.json"
    prod_path = "reports/ragas_report.json"
    naive = {}
    prod = {}

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as f:
            naive = json.load(f)
        with open(prod_path, encoding="utf-8") as f:
            prod = json.load(f)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
        print("-" * 55)
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            n = naive.get("aggregate", {}).get(m)
            p = prod.get("aggregate", {}).get(m)
            if n is None or p is None:
                print(f"  {m:<23} {'N/A':>8} {'N/A':>12} {'N/A':>8}")
                continue
            d = p - n
            status = "✓" if p >= 0.75 else " "
            print(f"{status} {m:<23} {n:>8.4f} {p:>12.4f} {d:>+8.4f}")

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    incomplete = [report for report in (naive, prod) if report.get("evaluation_status") != "complete"]
    if incomplete:
        print("\n⚠️ RAGAS chưa hoàn tất: " + prod_results.get("evaluation_error", "một số metric bị thiếu"))
        for report in incomplete:
            print(f"Provider/model: {report.get('evaluation_provider')}/{report.get('evaluation_model')}")
            print("Số mẫu đã chấm: " + str(report.get("metric_sample_counts", {})))
            if report.get("evaluation_error"):
                print(report["evaluation_error"])
        print("Các giá trị N/A/null không phải điểm RAGAS. Sau khi sửa lỗi, chạy python main.py --eval-only để chỉ chấm ô thiếu.")
    print("\n📋 Next steps:")
    print("  1. Điền analysis/failure_analysis.md")
    print("  2. Viết analysis/reflections/reflection_[HọTên].md")
    print("  3. Chạy: python check_lab.py")
    return 1 if incomplete else 0


if __name__ == "__main__":
    sys.exit(main(resume="--resume" in sys.argv[1:], eval_only="--eval-only" in sys.argv[1:]))
