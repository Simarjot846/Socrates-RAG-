FROM python:3.10-slim

# Install system dependencies (e.g. git, build-essential if needed for some libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy backend files
COPY . /code

# Create directories and set open permissions for HF user (UID 1000)
RUN mkdir -p /code/chroma_db /code/temp && chmod -R 777 /code

# Environment variables default overrides for HF Space
ENV CHROMA_DB_DIR=/code/chroma_db
ENV PORT=7860

# Run FastAPI on Hugging Face default port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
