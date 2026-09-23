# Python 3.13 slim image
FROM python:3.13.2-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

ENTRYPOINT ["python", "src/parcellation.py"]
CMD ["-i", "input", "-o", "output", "-m", "model"]
