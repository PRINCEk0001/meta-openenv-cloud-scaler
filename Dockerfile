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

EXPOSE 7860

# Runs inference directly, logging to stdout for OpenEnv evaluator
CMD ["python", "inference.py"]