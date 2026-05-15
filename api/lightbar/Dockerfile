FROM python:3.11-buster

WORKDIR /app

# Debian buster is EOL; its repos moved to archive.debian.org and the
# buster-updates suite was dropped. Point apt there so pigpio + i2c-tools
# install. Keeps the runtime identical to what produced b5942b.log.jsonl.
RUN sed -i \
        -e 's|deb.debian.org/debian-security|archive.debian.org/debian-security|g' \
        -e 's|deb.debian.org/debian|archive.debian.org/debian|g' \
        -e '/buster-updates/d' \
        /etc/apt/sources.list \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends pigpio i2c-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "pigpiod && exec python -m uvicorn app.main:app --host 0.0.0.0 --port 80"]
