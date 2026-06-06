FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY frontend.py .
COPY start.sh .
RUN chmod +x start.sh

RUN mkdir -p /data/uploads /data/chroma /data/db

ENV PYTHONUNBUFFERED=1

EXPOSE 10000

CMD ["bash", "start.sh"]
