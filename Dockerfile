FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml
COPY setup.py /app/setup.py
COPY README.md /app/README.md
COPY raganything /app/raganything
COPY examples /app/examples

RUN grep -v -E "^[[:space:]]*mineru" /app/requirements.txt > /tmp/requirements.no_mineru.txt \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements.no_mineru.txt \
    && python -m pip install --no-deps -e . \
    && python -m pip install \
        aiohttp \
        gradio \
        numpy \
        openai \
        paddleocr \
        pdfplumber \
        python-docx \
        pypdfium2 \
        pymupdf \
        reportlab \
        python-dotenv

EXPOSE 7860

CMD ["python", "examples/webui_gradio.py"]
