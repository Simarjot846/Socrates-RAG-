FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the fastembed ONNX model at BUILD time into a fixed cache dir
ENV FASTEMBED_CACHE_PATH=/app/fastembed_cache
RUN python -c "\
import os; \
os.environ['FASTEMBED_CACHE_PATH'] = '/app/fastembed_cache'; \
from fastembed import TextEmbedding; \
model = TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
list(model.embed(['warmup'])); \
print('Model cached successfully')"

COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
