FROM python:3.12-slim-bookworm

# Install uv the right way
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only dependency files first (cache-friendly)
COPY .python-version pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy app + model
COPY predict.py model_C=1.0.bin ./

EXPOSE 9696

# Run THROUGH uv
CMD ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
