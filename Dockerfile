FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY . .

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "python", "-m", "prepare_submission"]