FROM python:3.11-buster

WORKDIR /app

# Debian buster is EOL; its repos moved to archive.debian.org and the
# buster-updates suite was dropped. Point apt there so apt-get works.
# pigpio is only packaged in Raspbian (not Debian), so we build the
# daemon + library from source below. Keeps the runtime identical to
# what produced b5942b.log.jsonl.
RUN sed -i \
        -e 's|deb.debian.org/debian-security|archive.debian.org/debian-security|g' \
        -e 's|deb.debian.org/debian|archive.debian.org/debian|g' \
        -e '/buster-updates/d' \
        /etc/apt/sources.list \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends build-essential wget i2c-tools \
    && rm -rf /var/lib/apt/lists/*

# Build pigpiod + libpigpio from source (Debian buster doesn't ship pigpio).
RUN wget -q https://github.com/joan2937/pigpio/archive/refs/tags/v79.tar.gz \
    && tar xzf v79.tar.gz \
    && make -C pigpio-79 -j"$(nproc)" \
    && make -C pigpio-79 install \
    && ldconfig \
    && rm -rf pigpio-79 v79.tar.gz

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "pigpiod && exec python -m uvicorn app.main:app --host 0.0.0.0 --port 80"]
