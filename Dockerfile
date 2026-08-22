FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy requirements from backend directory
COPY ./backend/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy all files from the backend directory to the container's working directory
COPY ./backend /code

# Create directories and set permissions
RUN mkdir -p /code/chroma_db /code/temp && chmod -R 777 /code

# Set environment variables
ENV CHROMA_DB_DIR=/code/chroma_db

# Run the FastAPI server on port 10000 (Render default port is exposed via $PORT, uvicorn needs to bind to $PORT)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
