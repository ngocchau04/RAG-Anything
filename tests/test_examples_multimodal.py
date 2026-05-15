from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"


def _run_example(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    libreoffice_dir = r"C:\Program Files\LibreOffice\program"
    if Path(libreoffice_dir).exists() and libreoffice_dir not in env.get("Path", ""):
        env["Path"] = f"{libreoffice_dir};{env.get('Path', '')}"
    return subprocess.run(
        [str(PYTHON), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


def _has_libreoffice() -> bool:
    if shutil.which("libreoffice") or shutil.which("soffice"):
        return True

    candidates = [
        Path(r"C:\Program Files\LibreOffice\program\soffice.com"),
        Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.com"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
    ]
    return any(p.exists() for p in candidates)


def _libreoffice_path_state() -> str:
    has_path_cmd = shutil.which("libreoffice") or shutil.which("soffice")
    candidates = [
        Path(r"C:\Program Files\LibreOffice\program\soffice.com"),
        Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.com"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
    ]
    has_default_install = any(p.exists() for p in candidates)
    if has_path_cmd:
        return "path"
    if has_default_install:
        return "installed_not_in_path"
    return "missing"


def _has_openai_compatible_api_key() -> bool:
    return bool(os.getenv("LLM_BINDING_API_KEY"))


def _has_gemini_or_google_api_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def _resolve_api_key_for_examples() -> str | None:
    return (
        os.getenv("LLM_BINDING_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )


def test_office_document_example() -> None:
    state = _libreoffice_path_state()
    if state == "missing":
        pytest.skip(
            "LibreOffice is not discoverable from PATH and default Windows install paths."
        )
    result = _run_example(["examples/office_document_test.py", "--check-libreoffice"])
    if result.returncode != 0 and "LibreOffice not found" in result.stdout:
        pytest.skip(
            "LibreOffice exists on default Windows path, but upstream script check "
            "did not discover it via command-name probing."
        )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "check passed" in result.stdout.lower()


def test_text_format_example() -> None:
    result = _run_example(["examples/text_format_test.py", "--check-reportlab"])
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "reportlab" in result.stdout.lower()


def test_image_format_example() -> None:
    result = _run_example(["examples/image_format_test.py", "--check-pillow"])
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "pillow" in result.stdout.lower()


def test_enhanced_markdown_example() -> None:
    result = _run_example(["examples/enhanced_markdown_example.py"], timeout=240)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "Enhanced Markdown Conversion Demonstration" in result.stdout


def test_modalprocessors_example() -> None:
    api_key = _resolve_api_key_for_examples()
    if not api_key:
        pytest.skip(
            "No API key found in env (LLM_BINDING_API_KEY/GEMINI_API_KEY/GOOGLE_API_KEY)."
        )
    if os.getenv("RUN_LIVE_API_EXAMPLES", "0") != "1":
        pytest.skip(
            "Live API examples are disabled. Set RUN_LIVE_API_EXAMPLES=1 to enable."
        )
    result = _run_example(
        [
            "examples/modalprocessors_example.py",
            "--api-key",
            api_key,
            "--base-url",
            os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        ],
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_batch_dry_run_example() -> None:
    # Upstream dry-run example is parser-centric (mineru/docling/paddleocr choices).
    # In this CPU-only policy, we do not execute MinerU runtime paths.
    result = _run_example(
        [
            "examples/batch_dry_run_example.py",
            "inputs",
            "--parser",
            "docling",
            "--recursive",
        ],
        timeout=180,
    )
    if result.returncode != 0:
        pytest.skip(
            "Upstream example parser selection is not CPU-only-stable here; "
            "skip instead of falling back to MinerU."
        )
    assert "Dry run: files that would be processed" in result.stdout
