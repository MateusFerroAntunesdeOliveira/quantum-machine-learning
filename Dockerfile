FROM python:3.12

# Install "uv" in the global environment of the container
RUN pip install uv

WORKDIR /app

# Copy dependencies files and install dependencies
COPY pyproject.toml uv.lock* ./
RUN uv sync --system --no-cache

# Create logs directory e mark it as volume
RUN mkdir -p logs
VOLUME ["/app/logs"]

COPY src/ src/

WORKDIR /app/src

CMD ["python", "main.py"]
