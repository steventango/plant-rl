FROM python:3.11-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends pigpio i2c-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh", "-c", "pigpiod && exec python -m uvicorn app.main:app --host 0.0.0.0 --port 80"]
