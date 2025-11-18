# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# system deps (if needed for pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libjpeg-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app code and model
COPY . /app

EXPOSE 5000

ENV FLASK_APP=app.py

CMD ["python", "app.py"]
