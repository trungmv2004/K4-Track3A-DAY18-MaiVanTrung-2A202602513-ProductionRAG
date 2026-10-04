from __future__ import annotations

"""Single-call enrichment and deterministic fallbacks preserving source content."""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.llm import create_client, get_settings


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


def _llm(prompt: str, text: str, json_mode: bool = False) -> str:
    client = create_client(timeout=45)
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create(
        model=get_settings().model,
        temperature=0,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": prompt + " Chỉ dựa trên dữ liệu, không thêm thông tin."},
            {"role": "user", "content": text},
        ],
        **kwargs,
    )
    return (response.choices[0].message.content or "").strip()


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _fallback_metadata(text):
    lower = text.casefold()
    categories = {
        "it": ("mật khẩu", "vpn", "malware", "cntt"),
        "hr": ("nghỉ", "nhân viên", "bảo hiểm", "thử việc"),
        "finance": ("lương", "chi phí", "tạm ứng", "triệu"),
    }
    category = next((name for name, words in categories.items() if any(w in lower for w in words)), "policy")
    return {
        "topic": (_sentences(text) or ["general"])[0][:100],
        "entities": sorted(set(re.findall(r"\b[A-ZĐ]{2,}\b", text))),
        "date_range": sorted(set(re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text))),
        "category": category,
        "language": "vi",
    }


def summarize_chunk(text: str) -> str:
    if get_settings().api_key:
        try:
            summary = _llm("Tóm tắt đoạn văn trong tối đa 2 câu ngắn bằng tiếng Việt.", text)
            if summary:
                return summary
        except Exception:
            pass
    return " ".join(_sentences(text)[:2])


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    if n_questions <= 0:
        return []
    if get_settings().api_key:
        try:
            content = _llm(f"Tạo {n_questions} câu hỏi mà đoạn văn trả lời được, mỗi câu một dòng.", text)
            questions = [
                re.sub(r"^\s*[\d.\-)•]+\s*", "", s).strip() for s in content.splitlines() if s.strip()
            ]
            if questions:
                return questions[:n_questions]
        except Exception:
            pass
    return [f"Nội dung nào được quy định về: {s.rstrip('.!?')}?" for s in _sentences(text)[:n_questions]]


def contextual_prepend(text: str, document_title: str = "") -> str:
    prefix = f"Trích từ tài liệu {document_title}." if document_title else "Trích đoạn chính sách nội bộ."
    if get_settings().api_key:
        try:
            prefix = (
                _llm(
                    "Viết 1 câu mô tả vị trí và chủ đề đoạn văn trong tài liệu.",
                    f"Tài liệu: {document_title}\n\n{text}",
                )
                or prefix
            )
        except Exception:
            pass
    return f"{prefix}\n\n{text}"


def extract_metadata(text: str) -> dict:
    if get_settings().api_key:
        try:
            result = json.loads(
                _llm(
                    "Trả về JSON metadata: topic, entities (list), date_range (list), category, language.",
                    text,
                    True,
                )
            )
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    return _fallback_metadata(text)


def _enrich_single_call(text: str, source: str) -> dict:
    fallback = {
        "summary": " ".join(_sentences(text)[:2]),
        "questions": [f"Nội dung nào được quy định về: {s.rstrip('.!?')}?" for s in _sentences(text)[:3]],
        "context": f"Trích từ tài liệu {source}." if source else "Trích đoạn chính sách nội bộ.",
        "metadata": _fallback_metadata(text),
        "backend": "extractive",
    }
    if get_settings().api_key:
        try:
            result = json.loads(
                _llm(
                    "Trả về JSON gồm summary (2 câu), questions (3 câu hỏi), context (1 câu), "
                    "metadata (topic, entities, date_range, category, language).",
                    f"Tài liệu: {source}\n\n{text}",
                    True,
                )
            )
            if not isinstance(result, dict) or not all(
                isinstance(result.get(k), str) for k in ("summary", "context")
            ):
                raise ValueError("Invalid enrichment text fields")
            if not isinstance(result.get("questions"), list) or not all(
                isinstance(q, str) for q in result["questions"]
            ):
                raise ValueError("Invalid enrichment questions")
            if not isinstance(result.get("metadata"), dict):
                raise ValueError("Invalid enrichment metadata")
            return {**result, "backend": get_settings().provider}
        except Exception:
            return {**fallback, "backend": "extractive_after_api_error"}
    return fallback


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    methods = ["combined"] if methods is None else methods
    if set(methods) - {"combined", "summary", "hyqa", "contextual", "metadata"}:
        raise ValueError("Unknown enrichment method")
    enriched = []
    for i, chunk in enumerate(chunks):
        text, metadata = chunk["text"], dict(chunk.get("metadata", {}))
        source = metadata.get("source", "")
        if "combined" in methods:
            result = _enrich_single_call(text, source)
            summary, questions = result["summary"], result["questions"]
            base = f"{result['context']}\n\n{text}"
            auto_meta = result["metadata"]
            metadata["enrichment_backend"] = result["backend"]
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            base = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}
            metadata["enrichment_backend"] = (
                get_settings().provider + "_or_fallback" if get_settings().api_key else "extractive"
            )
        extra = ([f"Tóm tắt: {summary}"] if summary else []) + (
            ["Câu hỏi: " + " ".join(questions)] if questions else []
        )
        enriched_text = base + ("\n\n" + "\n".join(extra) if extra else "")
        # Source provenance and hierarchy are authoritative, never LLM-overwritable.
        enriched.append(
            EnrichedChunk(
                text, enriched_text, summary, questions, {**auto_meta, **metadata}, "+".join(methods)
            )
        )
        if (i + 1) % 10 == 0 or i + 1 == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)
    return enriched
