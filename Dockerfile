FROM python:3.12-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir \
    essentia==2.1b6.dev1389 \
    flask==3.1.0 \
    gunicorn==23.0.0 \
    numpy==2.2.0

COPY analyzer.py .

EXPOSE 5030

CMD ["gunicorn", "-b", "0.0.0.0:5030", "-w", "2", "--timeout", "120", "analyzer:app"]
