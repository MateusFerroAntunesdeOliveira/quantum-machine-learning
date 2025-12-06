FROM python:3.12-slim

# Install "uv" in the global environment of the container
RUN pip install uv

WORKDIR /app

# Copy dependencies files and install dependencies
# We use --system because inside Docker we don't necessarily need a virtual venv,
# but keeping uv sync standard is fine.
COPY pyproject.toml uv.lock* ./
RUN uv sync --system --no-cache

# Create data directories to match config.py expectations
RUN mkdir -p data/input data/output

# Volume for persistence (Inputs, Outputs, Logs are all inside data/)
VOLUME ["/app/data"]

COPY src/ src/

# Set PYTHONPATH to include /app to "src" is importable as a module
ENV PYTHONPATH=/app

# Default command: Run Step 01 as a module
# Syntax: uv run python -m <module_path>
CMD ["uv", "run", "python", "-m", "src.steps.01_data_cleaning"]
