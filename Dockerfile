FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces runs as non-root user 1000
RUN useradd -m -u 1000 user
USER user

# OpenEnv Phase 1 expects the server on 7860
EXPOSE 7860

# CMD launches the FastAPI server, which now handles OpenEnv logging internally
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]