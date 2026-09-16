FROM python:3.11-slim

WORKDIR /app

COPY deploy/hf_ui/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app/__init__.py app/__init__.py
COPY app/frontend.py app/frontend.py
COPY app/models/ app/models/
COPY data/public_test_cases.json data/public_test_cases.json
COPY data/custom_test_cases.json data/custom_test_cases.json

ENV API_BASE_URL="" \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

EXPOSE 7860

CMD ["streamlit", "run", "app/frontend.py", "--server.address", "0.0.0.0", "--server.port", "7860", "--server.headless", "true", "--server.enableCORS", "false"]
