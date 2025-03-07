FROM python:3.11

RUN apt update && apt install -y ffmpeg libsdl2-dev swig

WORKDIR /app

COPY ./pyproject.toml /app/pyproject.toml
RUN pip install -e .
