FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the fastembed ONNX model at BUILD time (not runtime)
# This bakes the model into the image so no outbound network needed at startup
RUN python -c "from fastembed import TextEmbedding; model = TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); list(model.embed(['warmup']))"

COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
