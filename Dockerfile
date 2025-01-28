FROM python:3.11

RUN apt update && apt install -y swig libsdl2-dev

WORKDIR /app

COPY ./pyproject.toml /app/pyproject.toml
RUN pip install -e .
