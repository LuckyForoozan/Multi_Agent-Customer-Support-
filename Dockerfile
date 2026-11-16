# Use a lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the whole project into the container
COPY . /app

# Install system packages needed by faiss‑cpu (optional but safe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libopenblas-dev && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install all Python dependencies
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

# Expose the FastAPI port
EXPOSE 8080

# Run the API when the container starts
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]