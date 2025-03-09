FROM python:3.11-buster

WORKDIR /app

RUN apt update && apt install -y --no-install-recommends gnupg
RUN echo "deb http://archive.raspberrypi.org/debian/ buster main" > /etc/apt/sources.list.d/raspi.list \ && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 82B129927FA3303E

RUN apt update && apt install -y --no-install-recommends \
  python3-picamera2 \
  && apt-get clean \
  && apt-get autoremove \
  && rm -rf /var/cache/apt/archives/* \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
