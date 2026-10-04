"""Refresh evidence from the offline run; never invent RAGAS scores or make API calls."""

import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_report(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def write_report(name, value):
    (REPORTS / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def tokens(text):
    return set(re.findall(r"\w+", text.casefold()))


def main():
    baseline = read_report("naive_baseline_report.json")
    production = read_report("ragas_report.json")
    assert all(r["configuration"]["llm_provider"] == "offline" for r in (baseline, production))
    assert all(r["num_questions"] == 20 for r in (baseline, production))
    assert all(v is None for r in (baseline, production) for v in r["aggregate"].values())
    stamp = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat()

    lines = [
        "# Latency breakdown - Lab 18", "", f"Run recorded: {stamp}.",
        "", "Profile: lightweight, LLM_PROVIDER=offline; processing on this machine only.",
        "Embedding: " + production["configuration"]["embedding_model"] + ".",
        "Reranker: " + production["configuration"]["reranker_model"] + ".",
        "Generation/enrichment use extractive fallback; these are not LLM inference timings.",
        "Gemini key is configured but the user declined sending corpus data to Google.",
        "RAGAS is unavailable until a local generative LLM/server is installed.",
        "Evaluation timing measures the skipped attempt, not RAGAS judging.",
        "Index timing includes encoder loading as needed; baseline and standalone production",
        "were measured in separate processes, so compare these runs with that limitation.",
        "", "| Build step | Baseline (ms) | Production (ms) |", "|---|---:|---:|",
    ]
    for step in ("load_and_chunk_ms", "enrichment_ms", "index_ms", "reranker_load_ms"):
        row = []
        for report in (baseline, production):
            value = report["latency"]["build_ms"].get(step)
            row.append(f"{value:.2f}" if value is not None else "N/A")
        lines.append(f"| {step} | {row[0]} | {row[1]} |")
    lines += [
        "", "| Pipeline | Query step | Samples | Mean (ms) | p50 (ms) | p95 (ms) | Max (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, report in (("Baseline", baseline), ("Production", production)):
        for step in ("retrieval_ms", "rerank_ms", "generation_ms"):
            values = [row[step] for row in report["latency"]["per_query"] if step in row]
            if values:
                lines.append(
                    f"| {label} | {step} | {len(values)} | {statistics.mean(values):.2f} | "
                    f"{percentile(values, .5):.2f} | {percentile(values, .95):.2f} | {max(values):.2f} |"
                )
    lines += ["", "Raw timings and source traces are preserved in the two JSON reports.", ""]
    (REPORTS / "latency_report.md").write_text("\n".join(lines), encoding="utf-8")

    cases = []
    for index, row in enumerate(production["per_question"], 1):
        truth = tokens(row["ground_truth"])
        cases.append({
            "test_question_number": index, **{k: row[k] for k in (
                "question", "answer", "ground_truth", "contexts"
            )},
            "answer_token_overlap": len(tokens(row["answer"]) & truth) / max(1, len(truth)),
            "context_token_overlap": len(tokens("\n".join(row["contexts"])) & truth) / max(1, len(truth)),
            "selected_sources": production["latency"]["per_query"][index - 1]["selected_contexts"],
        })
    selected = sorted(cases, key=lambda row: row["answer_token_overlap"])[:5]
    write_report("diagnostic_review.json", {
        "selection_method": "ground_truth_token_overlap_for_manual_review", "not_ragas": True,
        "ragas_status": production["evaluation_status"], "configuration": production["configuration"],
        "cases": selected,
    })
    print("Manual review question numbers:", [row["test_question_number"] for row in selected])

    check_log = (ROOT / ".cache/offline_check_lab.log").read_text(encoding="utf-8")
    passed, total = map(int, re.search(r"(\d+)/(\d+) tests passed", check_log).groups())
    exit_code = int((ROOT / ".cache/offline_check_lab.exit").read_text())
    write_report("validation_report.json", {
        "recorded_at": stamp, "timezone": "Asia/Bangkok", "profile": "lightweight_offline",
        "models": production["configuration"],
        "pytest": {"passed": passed, "failed": total - passed, "original_tests": 37,
                   "regression_tests": 12, "provider_tests": 9},
        "main_exit_code": int((ROOT / ".cache/offline_main.exit").read_text()),
        "standalone_pipeline_exit_code": int((ROOT / ".cache/offline_pipeline.exit").read_text()),
        "ragas_status": production["evaluation_status"], "ragas_error": production["evaluation_error"],
        "remaining_requirement": "Install a local generative LLM/server, run real RAGAS on 20 questions "
                                 "for baseline/production, then replace proxy review with RAGAS bottom-5.",
        "cloud_corpus_export_authorized": False, "cloud_corpus_export_performed": False,
        "gemini_key_configured": True, "local_generative_llm_available": False,
        "commit_performed": False, "push_performed": False,
        "check_lab_exit_code": exit_code,
        "check_lab_remaining_errors": [production["evaluation_error"]] if exit_code else [],
        "ruff_check": "passed" if int((ROOT / ".cache/offline_ruff.exit").read_text()) == 0 else "failed",
        "remaining_todos": 0,
        "not_ready_for_submission": exit_code != 0,
    })
    print(f"Reports refreshed; tests={passed}/{total}, check_lab exit={exit_code}.")


if __name__ == "__main__":
    main()
