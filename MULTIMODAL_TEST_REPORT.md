# MULTIMODAL TEST REPORT

## Summary

- Environment: Windows CPU-only, no GPU/CUDA.
- Policy: MinerU runtime is intentionally disabled in this validation flow.
- LibreOffice: installed and discoverable from PATH via `soffice.com/soffice.exe`.
- Current results:
  - `pytest -q tests/test_examples_multimodal.py tests/test_multimodal_modalities.py -rs` -> `15 passed, 1 skipped`
  - `pytest -q -rs` -> `203 passed, 2 skipped`

## LibreOffice

- `examples/office_document_test.py` detection was updated to:
  - probe `soffice.com`, `soffice`, `libreoffice` with `shutil.which`
  - fallback to default Windows install paths:
    - `C:\Program Files\LibreOffice\program\soffice.com`
    - `C:\Program Files\LibreOffice\program\soffice.exe`
    - `C:\Program Files (x86)\LibreOffice\program\soffice.com`
    - `C:\Program Files (x86)\LibreOffice\program\soffice.exe`
- `test_office_document_example` now passes in CPU-only environment.

## MinerU

- Upstream code still contains MinerU references/defaults.
- In this repo test flow, MinerU is not executed.
- If a path is MinerU/GPU-bound, it is skipped with explicit CPU-only reason.

## Gemini / modalprocessors

- `examples/modalprocessors_example.py` expects CLI `--api-key`.
- Test resolves key from `LLM_BINDING_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY`.
- Live API call remains opt-in (`RUN_LIVE_API_EXAMPLES=1`), so this test is skipped by default.

## Evidence Matrix

| File type | Text | Image | Table | Equation | Chart | Evidence level | Evidence test | Status | Notes |
| --------- | ---- | ----- | ----- | -------- | ----- | -------------- | ------------- | ------ | ----- |
| `.docx` | Yes | Yes (caption marker) | Yes | Yes (formula marker) | Yes (caption marker) | `parser_output` | `test_docx_multimodal_extraction`, `test_text_extraction`, `test_image_extraction`, `test_table_extraction`, `test_equation_extraction`, `test_chart_extraction` | Passed | Parsed via `simple_docx` |
| `.pdf` | Yes (marker text) | Yes (marker text) | Yes (marker text) | Yes (marker text) | Yes (marker text) | `parser_output` | `test_pdf_multimodal_extraction` | Passed | Parsed via `docling` in CPU-only mode |
| `.pptx` | Yes (marker text) | Yes (marker text) | Yes (marker text) | Yes (marker text) | Yes (marker text) | `parser_output` | `test_pptx_multimodal_extraction` | Passed | Parsed via `docling` + office conversion path |
| `.md` | Yes | Yes | Yes | Yes | Yes | `fixture` | `test_md_multimodal_fixture_markers` | Passed | Fixture content assertion, not parser runtime |
| `.html` | Yes | Yes | Yes | Yes | Yes | `fixture` | `test_html_multimodal_fixture_markers` | Passed | Fixture content assertion, not parser runtime |

## Remaining skips

1. `tests/test_examples_multimodal.py::test_modalprocessors_example`
- Reason: live API tests disabled by default (`RUN_LIVE_API_EXAMPLES=1` required).

2. `tests/test_core_modules.py::test_patch_hf_symlink_fallback_copy_windows`
- Reason: Windows symlink privilege not available in this environment.

## Notes on evidence level

- `fixture`: validates marker presence in fixture file content only.
- `parser_output`: validates markers are extractable from parser runtime output.
- `index_retrieval`: not added in this phase for these fixture tests; index/query flow is covered by separate pipeline tests/examples.
