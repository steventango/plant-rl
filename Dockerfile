FROM python:3.11

RUN apt update && apt install -y ffmpeg libsdl2-dev swig

COPY --from=ghcr.io/astral-sh/uv:0.7.14 /uv /uvx /bin/

WORKDIR /app

COPY ./pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
RUN uv sync --dev --prerelease=allow
