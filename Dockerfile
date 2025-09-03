# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir datasets nltk tqdm unidecode
RUN pip install sentence-transformers faiss-cpu torch

# Pre-fetch NLTK models so the container can run offline
RUN python - << 'PY'
import nltk
for pkg in ["stopwords","punkt","wordnet","omw-1.4","averaged_perceptron_tagger_eng"]:
    nltk.download(pkg)
PY

# Code is volume-mounted by compose; COPY is optional
COPY src /app/src