FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["LIGHTNING_USER_ID=abc", "LIGHTNING_API_KEY=123", "lightning", "deploy", "app/main.py"]
