FROM python:3.10-slim

# Install system dependencies
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

# Create directories and set open permissions
RUN mkdir -p /code/chroma_db /code/temp && chmod -R 777 /code

# Set environment variables
ENV CHROMA_DB_DIR=/code/chroma_db

# Run FastAPI using dynamic port binding (important for Render)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
