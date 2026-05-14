FROM python:3.11

RUN apt update && apt install -y ffmpeg libsdl2-dev swig sudo

COPY --from=ghcr.io/astral-sh/uv:0.7.14 /uv /uvx /bin/

ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /app
RUN chown $USERNAME:$USERNAME /app

USER $USERNAME

COPY --chown=$USERNAME:$USERNAME ./pyproject.toml /app/pyproject.toml
COPY --chown=$USERNAME:$USERNAME uv.lock /app/uv.lock
RUN uv sync --dev --prerelease=allow
