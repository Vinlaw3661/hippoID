FROM --platform=arm64 python:3.11.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl cmake && rm -rf /var/lib/apt/lists/*

# Transfer files to the container
COPY setup-checks.sh .
COPY src/ .
COPY pyproject.toml .

# Install project dependencies
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv install --system

# Set up runtime directories
RUN mkdir /app/runtime/audio && \
    mkdir /app/runtime/faces

# Run setup checks
RUN chmod +x ./setup-checks.sh && ./setup-checks.sh

CMD ["python", "hippo_id/main.py"]
