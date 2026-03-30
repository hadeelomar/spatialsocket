FROM python:3.9-slim

WORKDIR /app

# system deps for scipy/numpy and soundcard
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libsndfile1 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source
COPY app/ ./app/
COPY http_server/ ./http_server/
COPY hrtf_datasets/ ./hrtf_datasets/

# runtime dirs
RUN mkdir -p hrtf_uploads uploads app/uploads http_server/uploads app/rendered rendered

ENV HRTF_DATASETS_DIR=/app/hrtf_datasets
ENV WS_HOST=0.0.0.0
ENV WS_PORT=5000
ENV HTTP_HOST=0.0.0.0
ENV HTTP_PORT=5001
ENV FLASK_ENV=production

EXPOSE 5000 5001

# ws server: --chdir into app/ so bare imports (config, session_manager etc) resolve
# http server: runs from /app so http_server.main:app works; sys.path insert in main.py covers app.*
CMD gunicorn \
      --chdir /app/http_server \
      --worker-class gevent \
      --workers 1 \
      --bind "${HTTP_HOST}:${HTTP_PORT}" \
      --timeout 300 \
      --log-level info \
      --name spatialsocket-http \
      "main:app" & \
    gunicorn \
      --chdir /app/app \
      --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
      --workers 1 \
      --bind "${WS_HOST}:${WS_PORT}" \
      --timeout 300 \
      --log-level info \
      --name spatialsocket-ws \
      "main:app"
