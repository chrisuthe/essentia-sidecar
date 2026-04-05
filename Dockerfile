FROM python:3.12-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir \
    essentia==2.1b6.dev1389 \
    flask==3.1.0 \
    gunicorn==23.0.0 \
    numpy==2.2.0

COPY analyzer.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

EXPOSE 5030

ENTRYPOINT ["./entrypoint.sh"]
