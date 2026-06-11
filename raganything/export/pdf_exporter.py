from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


DEFAULT_LINUX_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
DEFAULT_LINUX_BOLD_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
DEFAULT_WINDOWS_FONT = Path(r"C:\Windows\Fonts\arial.ttf")
DEFAULT_WINDOWS_BOLD_FONT = Path(r"C:\Windows\Fonts\arialbd.ttf")
FONT_REGULAR_NAME = "RAGAnythingUnicode"
FONT_BOLD_NAME = "RAGAnythingUnicodeBold"


def _safe_stem(prefix: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", prefix).strip("_") or "report"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_font_path(env_name: str, candidates: list[Path]) -> Path:
    override = (os.getenv(env_name) or "").strip()
    if override:
        p = Path(override)
        if p.exists():
            return p
    for p in candidates:
        if p.exists():
            return p
    raise RuntimeError(
        "Unicode font not found. Please install DejaVuSans or configure "
        "PDF_EXPORT_FONT_PATH."
    )


def _register_unicode_fonts() -> tuple[str, str]:
    regular_path = _resolve_font_path(
        "PDF_EXPORT_FONT_PATH",
        [DEFAULT_LINUX_FONT, DEFAULT_WINDOWS_FONT],
    )
    bold_path = _resolve_font_path(
        "PDF_EXPORT_BOLD_FONT_PATH",
        [DEFAULT_LINUX_BOLD_FONT, DEFAULT_WINDOWS_BOLD_FONT],
    )
    if FONT_REGULAR_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_REGULAR_NAME, str(regular_path)))
    if FONT_BOLD_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_BOLD_NAME, str(bold_path)))
    return FONT_REGULAR_NAME, FONT_BOLD_NAME


def _clean_markdown_text(text: str) -> str:
    if not text:
        return ""
    out = str(text).replace("\r\n", "\n").replace("\r", "\n")
    out = re.sub(r"^#{1,6}\s*", "", out, flags=re.MULTILINE)
    out = re.sub(r"\*\*(.*?)\*\*", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"__(.*?)__", r"\1", out, flags=re.DOTALL)
    out = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", out)
    out = re.sub(r"`([^`]+)`", r"\1", out)
    return out.strip()


def _parse_markdown_sections(content: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading: Optional[str] = None
    current_lines: list[str] = []
    for raw_line in (
        str(content or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    ):
        line = raw_line.rstrip()
        if line.startswith("# "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[2:].strip()
            current_lines = []
            continue
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip()
            current_lines = []
            continue
        current_lines.append(line)
    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return sections


def _wrap_text(text: str, max_chars: int = 95) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    for paragraph in str(text).splitlines():
        p = paragraph.strip()
        if not p:
            lines.append("")
            continue
        words = p.split()
        cur: list[str] = []
        cur_len = 0
        for w in words:
            add_len = len(w) + (1 if cur else 0)
            if cur and cur_len + add_len > max_chars:
                lines.append(" ".join(cur))
                cur = [w]
                cur_len = len(w)
            else:
                cur.append(w)
                cur_len += add_len
        if cur:
            lines.append(" ".join(cur))
    return lines


def _is_markdown_table_row(line: str) -> bool:
    s = (line or "").strip()
    return s.startswith("|") and s.endswith("|") and s.count("|") >= 2


def _render_lines_for_pdf(content: str, max_chars: int = 95) -> list[str]:
    out_lines: list[str] = []
    for raw in _clean_markdown_text(content).splitlines():
        line = raw.rstrip()
        if not line:
            out_lines.append("")
            continue
        # Keep table rows as-is to avoid breaking markdown table layout.
        if _is_markdown_table_row(line):
            out_lines.append(line)
            continue
        # Keep equation lines intact when possible.
        if "=" in line and any(
            k in line.lower() for k in ["tp", "tn", "fp", "fn", "accuracy", "roc"]
        ):
            out_lines.extend(_wrap_text(line, max_chars=max_chars))
            continue
        out_lines.extend(_wrap_text(line, max_chars=max_chars))
    return out_lines


def _write_pdf(
    title: str,
    sections: list[tuple[str, str]],
    output_path: Path,
    *,
    title_page_lines: Optional[list[str]] = None,
    toc_lines: Optional[list[str]] = None,
) -> Path:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    regular_font, bold_font = _register_unicode_fonts()
    c = canvas.Canvas(str(output_path), pagesize=A4)
    width, height = A4
    x = 48
    y = height - 52

    def new_page() -> None:
        nonlocal y
        c.showPage()
        y = height - 52

    c.setFont(bold_font, 18)
    c.drawString(x, y, title)
    y -= 28

    if title_page_lines:
        c.setFont(regular_font, 11)
        for line in title_page_lines:
            for wrapped in _wrap_text(line, max_chars=90):
                if y < 70:
                    new_page()
                    c.setFont(regular_font, 11)
                c.drawString(x, y, wrapped)
                y -= 15
        new_page()
        c.setFont(bold_font, 14)
        c.drawString(x, y, "Table of Contents")
        y -= 22
        c.setFont(regular_font, 10)
        for line in toc_lines or []:
            if y < 70:
                new_page()
                c.setFont(regular_font, 10)
            c.drawString(x, y, line)
            y -= 13
        new_page()

    for heading, content in sections:
        if y < 90:
            new_page()
        c.setFont(bold_font, 11)
        c.drawString(x, y, heading)
        y -= 16
        c.setFont(regular_font, 10)
        for line in _render_lines_for_pdf(content, max_chars=95):
            if y < 60:
                new_page()
                c.setFont(regular_font, 10)
            c.drawString(x, y, line)
            y -= 13
        y -= 8

    c.save()
    return output_path


def export_current_answer_to_pdf(
    question: str,
    answer: str,
    source_file: Optional[str],
    output_dir: str | Path,
) -> Path:
    out_dir = _ensure_dir(Path(output_dir))
    filename = f"{_safe_stem('answer_report')}_{_timestamp()}.pdf"
    output_path = out_dir / filename
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = [
        ("Generated time", now),
        ("Source file", source_file or "N/A"),
        ("User question", question or "N/A"),
        ("Assistant answer", answer or "N/A"),
    ]
    return _write_pdf("RAG-Anything Answer Report", sections, output_path)


def export_chat_history_to_pdf(
    messages: list[dict],
    selected_file: Optional[str],
    output_dir: str | Path,
) -> Path:
    out_dir = _ensure_dir(Path(output_dir))
    filename = f"{_safe_stem('chat_history')}_{_timestamp()}.pdf"
    output_path = out_dir / filename
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    convo_lines: list[str] = []
    for m in messages or []:
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        if not content:
            continue
        prefix = "User" if role == "user" else "Assistant"
        convo_lines.append(f"{prefix}: {content}")
    conversation = "\n".join(convo_lines) if convo_lines else "N/A"

    sections = [
        ("Generated time", now),
        ("Indexed/selected file", selected_file or "N/A"),
        ("Conversation history", conversation),
    ]
    return _write_pdf("RAG-Anything Chat History", sections, output_path)


def export_structured_report_to_pdf(
    report_title: str,
    source_file: Optional[str],
    source_doc_id: Optional[str],
    user_request: str,
    report_body: str,
    output_dir: str | Path,
    filename_prefix: str = "rag_report",
    source_validation_status: Optional[str] = None,
    sections_generated: Optional[list[str]] = None,
    report_mode: Optional[str] = None,
    inferred_report_type: Optional[str] = None,
) -> Path:
    out_dir = _ensure_dir(Path(output_dir))
    filename = f"{_safe_stem(filename_prefix)}_{_timestamp()}.pdf"
    output_path = out_dir / filename
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parsed_sections = _parse_markdown_sections(report_body or "")
    if not parsed_sections:
        parsed_sections = [("Report", report_body or "N/A")]
    title_page_lines = [
        f"Source document: {source_file or 'N/A'}",
        f"Source document id: {source_doc_id or 'N/A'}",
        f"Generated time: {now}",
        f"Report request: {user_request or 'N/A'}",
        f"Inferred report type: {inferred_report_type or 'custom'}",
        f"Source validation status: {source_validation_status or 'unknown'}",
        f"Report mode: {report_mode or 'structured'}",
    ]
    toc_lines = sections_generated or [heading for heading, _ in parsed_sections]
    return _write_pdf(
        report_title or "RAG-Anything Report",
        parsed_sections,
        output_path,
        title_page_lines=title_page_lines,
        toc_lines=toc_lines,
    )
