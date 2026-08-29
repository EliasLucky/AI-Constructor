FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONNUNBUFFERED 1

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	libpq-dev \
	&& rm -rf /var/lib/apt-get/lists/*

WORKDIR /APP

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/
