# Use official Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all your project files into the container
COPY . /app

# Install system dependencies (if any needed for torch, e.g., build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Command to run the app with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
