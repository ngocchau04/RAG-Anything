from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from .pdf_exporter import export_structured_report_to_pdf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def infer_language(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return "en"
    if re.search(r"[ăâđêôơưĂÂĐÊÔƠƯ]", raw):
        return "vi"
    lowered = raw.lower()
    if any(
        token in lowered for token in ["tóm tắt", "báo cáo", "tiếng việt", "tài liệu"]
    ):
        return "vi"
    return "en"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _strip_source_reference_lines(text: str) -> str:
    cleaned_lines: list[str] = []
    skip_refs = False
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        lowered = line.lower()
        if lowered.startswith("source:") or lowered.startswith("source file:"):
            continue
        if lowered in {
            "references",
            "references / source notes",
            "reference",
            "ghi chú nguồn",
        }:
            skip_refs = True
            continue
        if skip_refs:
            if (
                re.match(r"^[-*]\s*\[\d+\]\s+", line)
                or re.match(r"^\[\d+\]\s+", line)
                or line.startswith("- ")
            ):
                continue
            skip_refs = False
        cleaned_lines.append(raw_line)
    return "\n".join(cleaned_lines).strip()


def _dedupe_paragraphs(text: str) -> str:
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for paragraph in paragraphs:
        normalized = _normalize_text(paragraph)
        if normalized in seen:
            continue
        if deduped:
            prev = _normalize_text(deduped[-1])
            if normalized in prev or prev in normalized:
                continue
        seen.add(normalized)
        deduped.append(paragraph)
    return "\n\n".join(deduped)


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [c.strip() for c in chunks if c.strip()]


def _summary_from_text(text: str, max_sentences: int = 4) -> str:
    sentences = _split_sentences(_dedupe_paragraphs(text))
    return " ".join(sentences[:max_sentences]).strip()


def _key_points_from_text(text: str, max_points: int = 5) -> str:
    points: list[str] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        line = re.sub(r"^[-*]\s*", "", line).strip()
        if not line:
            continue
        normalized = _normalize_text(line)
        if normalized in seen:
            continue
        seen.add(normalized)
        points.append(f"- {line}")
        if len(points) >= max_points:
            break
    if not points:
        for sentence in _split_sentences(text)[:max_points]:
            points.append(f"- {sentence}")
    return "\n".join(points).strip()


def _remove_overlap(base: str, extra: str) -> str:
    base_norm = {_normalize_text(s) for s in _split_sentences(base)}
    kept: list[str] = []
    for sentence in _split_sentences(extra):
        if _normalize_text(sentence) in base_norm:
            continue
        kept.append(sentence)
    return " ".join(kept).strip()


def build_report_markdown(
    *,
    request: str,
    source_file: str,
    summary: str,
    key_points: str,
    details: str,
    limitations: str,
    language: str,
    image_mode: bool = False,
) -> str:
    if image_mode:
        if language == "vi":
            return (
                "# Báo cáo hình ảnh\n\n"
                "## Tóm tắt ngắn\n"
                f"{summary}\n\n"
                "## Chi tiết hình ảnh chính\n"
                f"{details}\n\n"
                "## Ghi chú / hạn chế\n"
                f"{limitations}\n\n"
                "## Ghi chú nguồn\n"
                f"- Source file: {source_file}\n"
                f"- Request: {request}\n"
            )
        return (
            "# Image Report\n\n"
            "## Short summary\n"
            f"{summary}\n\n"
            "## Main visual details\n"
            f"{details}\n\n"
            "## Notes / limitations\n"
            f"{limitations}\n\n"
            "## References / source notes\n"
            f"- Source file: {source_file}\n"
            f"- Request: {request}\n"
        )

    if language == "vi":
        return (
            "# Báo cáo tài liệu\n\n"
            "## Tóm tắt điều hành\n"
            f"{summary}\n\n"
            "## Các điểm chính\n"
            f"{key_points}\n\n"
            "## Phân tích chi tiết\n"
            f"{details}\n\n"
            "## Bảng / Công thức / Hình ảnh quan trọng\n"
            "Thông tin được tổng hợp từ chỉ mục hiện có của tài liệu.\n\n"
            "## Hạn chế\n"
            f"{limitations}\n\n"
            "## Ghi chú nguồn\n"
            f"- Source file: {source_file}\n"
            f"- Request: {request}\n"
        )
    return (
        "# Document Report\n\n"
        "## Executive summary\n"
        f"{summary}\n\n"
        "## Key points\n"
        f"{key_points}\n\n"
        "## Detailed analysis\n"
        f"{details}\n\n"
        "## Important tables / equations / figures\n"
        "The report uses only information available in the current index.\n\n"
        "## Limitations\n"
        f"{limitations}\n\n"
        "## References / source notes\n"
        f"- Source file: {source_file}\n"
        f"- Request: {request}\n"
    )


def generate_pdf_report_from_index(
    request: str,
    doc_record: Any,
    output_dir: str | Path,
    query_func: Callable[[str, Any], str],
) -> Path:
    req = (request or "").strip()
    if not req:
        raise ValueError("Please enter a report request.")
    source_file = str(getattr(doc_record, "original_filename", "") or "N/A")
    source_ext = Path(source_file).suffix.lower()
    image_mode = source_ext in IMAGE_EXTENSIONS
    language = infer_language(req)

    synthesis_q = (
        f"Create a concise report draft for this request: {req}. Include summary, key points, details, and limitations."
        if language != "vi"
        else f"Tạo bản nháp báo cáo ngắn gọn cho yêu cầu: {req}. Bao gồm tóm tắt, điểm chính, chi tiết và hạn chế."
    )
    limitation_q = (
        "List only limitations or missing information for this report."
        if language != "vi"
        else "Chỉ liệt kê các hạn chế hoặc thông tin còn thiếu của báo cáo."
    )

    raw_synthesis = _strip_source_reference_lines(
        str(query_func(synthesis_q, doc_record) or "").strip()
    )
    raw_limitations = _strip_source_reference_lines(
        str(query_func(limitation_q, doc_record) or "").strip()
    )
    combined = _dedupe_paragraphs(raw_synthesis)

    summary = _summary_from_text(combined, max_sentences=3 if image_mode else 4)
    key_points = _key_points_from_text(combined)
    details = _remove_overlap(summary + "\n" + key_points, combined)
    limitations = _dedupe_paragraphs(raw_limitations)

    if not limitations:
        limitations = (
            "Visual analysis was not available in the current index."
            if language != "vi"
            else "Phân tích hình ảnh có thể chưa khả dụng trong chỉ mục hiện tại."
        )
    if not summary:
        summary = (
            "No summary generated." if language != "vi" else "Không tạo được tóm tắt."
        )
    if not details:
        details = (
            "No additional details generated."
            if language != "vi"
            else "Không tạo được chi tiết bổ sung."
        )
    if image_mode:
        key_points = ""

    report_md = build_report_markdown(
        request=req,
        source_file=source_file,
        summary=summary,
        key_points=key_points
        or (
            "No key points generated."
            if language != "vi"
            else "Không tạo được điểm chính."
        ),
        details=details,
        limitations=limitations,
        language=language,
        image_mode=image_mode,
    )

    title = (
        "RAG-Anything Generated Report"
        if language != "vi"
        else "Báo cáo tạo tự động RAG-Anything"
    )
    return export_structured_report_to_pdf(
        report_title=title,
        source_file=source_file,
        user_request=req,
        report_body=report_md,
        output_dir=output_dir,
        filename_prefix="rag_report",
    )
