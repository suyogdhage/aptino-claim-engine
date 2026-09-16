FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY data/ data/

ENV POLICY_PDF_PATH=./data/policy.pdf \
    CHROMA_PERSIST_DIR=./chroma_db \
    HF_HOME=./hf_cache \
    FORCE_MOCK_LLM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
