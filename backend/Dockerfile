FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the fastembed ONNX model at BUILD time into home cache
# fastembed defaults to ~/.cache/fastembed — set HOME so it's predictable
ENV HOME=/app
RUN python -c "\
from fastembed import TextEmbedding; \
model = TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
list(model.embed(['warmup'])); \
print('Model cached at ~/.cache/fastembed')"

# Verify the cache exists
RUN ls -la /app/.cache/fastembed/ || echo "Cache dir not found"

COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
