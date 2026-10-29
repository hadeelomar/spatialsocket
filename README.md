# Comparison of Protocols for Server-Side Binaural Audio Rendering on the Web 

> Student: Hadeel Mohamed Said Omar
> Supervisor: Dr. Johan Pauwels

This repository contains the implementation for my final year undergraduate project.

The project studies whether HRTF-based spatial audio can be processed on the server, rather than in the browser, while still meeting the timing requirements of interactive applications.

The work focuses on three main questions:

- whether server-side spatial audio is practical for real-time use
- whether WebSocket or HTTP is better suited to this workload
- how well the system scales as the number of concurrent users increases

The repository includes:

- a WebSocket API for real-time spatial audio streaming
- an HTTP + SSE version with the same core features for comparison
- a browser demo and benchmark page
- unit tests and integration tests
- Locust load testing for concurrency and scalability checks

Audio rendering is handled by `py3dti`, which applies HRTF-based binaural processing to mono audio and produces stereo output for playback.

## Project Aim

The system allows a client to upload mono audio, place one or more virtual sound sources in 3D space, and receive processed binaural audio back from the server in real time.

This is useful as a research prototype because it moves the spatial processing away from the client device. That makes it possible to investigate centralised HRTF management, more consistent rendering behaviour, and the trade-off between network overhead and audio-processing cost.

The main workflow is:

1. create a session
2. initialise an HRTF renderer
3. create one or more audio sources
4. upload mono audio files
5. start streaming processed binaural audio
6. update source positions in real time
7. measure latency and scaling behaviour

Each client session gets its own spatial audio processor. This allows the server to track source positions separately for each user and render audio independently per session.

The default configuration uses:

- sample rate: `48000`
- buffer size: `1024`
- default HRTF file: `KEMAR_GRAS_EarSim_SmallEars_FreeFieldComp_48kHz.sofa`

## System Structure

### `app/`

This is the main WebSocket implementation.

- `main.py` starts the app, defines the Socket.IO handlers, and runs the renderer pool and render loop
- `audio_processor.py` wraps the `py3dti` processing logic
- `session_manager.py` stores and manages active sessions
- `source_manager.py` handles uploaded audio and source state
- `config.py` stores environment-based configuration

### `http_server/`

This is the HTTP comparison implementation.

It exposes REST endpoints for setup and control, and uses Server-Sent Events to stream audio back to the client.

### `tests/`

This folder contains:

- unit tests
- integration tests
- load testing with Locust

### `index.html`

This is the browser client used for demonstration and benchmarking.

It can be used to:

- test the WebSocket API
- test the HTTP API
- move sources in 3D space
- run the browser benchmark

## Architecture Summary

The WebSocket server runs with `gunicorn`, `gevent`, and `Flask-SocketIO`. The HTTP comparison server uses the same core processing logic, but exposes REST endpoints and Server-Sent Events instead of Socket.IO.

The general flow is:

```text
client
  |
  | WebSocket or HTTP
  |
server endpoint
  |
session manager
  |
audio processor (py3dti)
  |
binaural audio chunks returned to client
```

The system also supports a renderer pool. Renderers can be preloaded at startup so that session creation does not always need to wait for HRTF initialisation, which is one of the more expensive setup steps.

## Requirements

You need:

- Python `3.9+`
- `libsndfile`
- the Python packages in `requirements.txt`

If you want to use the container-based setup, you also need:

- Docker
- Docker Compose

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you also want testing and load-testing tools:

```bash
pip install -r requirements-dev.txt
```

### 3. Check HRTF files

Make sure the SOFA files you want to use are inside `hrtf_datasets/`.

If you need alternative HRTFs for testing, check the `hrtf_datasets/` folder first, as it already contains some additional options beyond the default file.

If you want to try more HRTFs, the [SONICOM repository](https://transfer.ic.ac.uk:9090/#/2022_SONICOM-HRTF-DATASET/) is a good place to look for extra datasets.

The default file used in the project is `KEMAR_GRAS_EarSim_SmallEars_FreeFieldComp_48kHz.sofa`.

## Running the Project Locally

### Start the WebSocket server

From the project root:

```bash
gunicorn \
  --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
  --workers 1 \
  --bind 0.0.0.0:5000 \
  --timeout 300 \
  --chdir app \
  main:app
```

### Start the HTTP server

In another terminal:

```bash
gunicorn \
  --worker-class gevent \
  --workers 1 \
  --bind 0.0.0.0:5001 \
  --timeout 300 \
  --chdir http_server \
  main:app
```

### Open the demo page

Open `index.html` in a browser.

Use:

- `http://localhost:5000` for the WebSocket server
- `http://localhost:5001` for the HTTP server

## Running with Docker

```bash
docker compose up --build
```

This starts:

- WebSocket server on port `5000`
- HTTP server on port `5001`

## Health Check

You can check whether the server is up and whether the renderer pool has filled:

```bash
curl http://localhost:5000/health
```

Example response:

```json
{"status":"healthy","renderer_pool_available":84,"renderer_pool_size":100}
```

## Configuration

These environment variables can be used to change the runtime setup:

| Variable | Default | What it does |
|---|---|---|
| `RENDERER_POOL_SIZE` | `0` | Number of renderers to preload at startup |
| `BUFFER_SIZE` | `1024` | Number of samples rendered per chunk |
| `SAMPLE_RATE` | `48000` | Audio sample rate |
| `MAX_SOURCES_PER_SESSION` | `16` | Max number of simultaneous sources |
| `SECRET_KEY` | none | Flask secret key |
| `HRTF_DATASETS_DIR` | `./hrtf_datasets` | Folder containing SOFA files |

## WebSocket API

The main WebSocket workflow is:

1. connect to the Socket.IO server
2. send `init_audio`
3. send `create_source`
4. upload a mono audio file over HTTP
5. send `stream_uploaded_file`
6. send `update_position` as needed

### Main WebSocket events

| Event sent by client | Example payload | Expected response |
|---|---|---|
| `init_audio` | `{}` or `{"hrtf_file":"KEMAR_GRAS_EarSim_SmallEars_FreeFieldComp_48kHz.sofa"}` | `status` with `AUDIO_INITIALISED` |
| `create_source` | `{"source_id":"src1","position":{"x":1,"y":0,"z":0}}` | `status` with `SOURCE_CREATED` |
| `stream_uploaded_file` | `{"source_id":"src1","filename":"file.wav"}` | `file_audio_chunk` events |
| `update_position` | `{"source_id":"src1","position":{"x":2,"y":0,"z":-1}}` | `status` with `POSITION_UPDATED` |
| `stop_file_stream` | `{}` | `status` with `STREAM_STOPPED` |

### Upload endpoint used by WebSocket flow

```text
POST /upload/<session_id>/<source_id>
Content-Type: multipart/form-data
field: audio_file
```

### Audio chunk format

Each `file_audio_chunk` event contains audio data plus some session information.

Example:

```json
{
  "audio_data": "<base64 interleaved stereo PCM>",
  "source_positions": {"src1": {"current": 42, "total": 300}},
  "render_ts": 1743280000000
}
```

The audio is base64-encoded stereo PCM using 32-bit float samples. Each chunk represents server-rendered binaural audio for the current session state.

## HTTP + SSE API

The HTTP version follows the same logical flow as the WebSocket implementation, but uses REST endpoints for control and Server-Sent Events for audio delivery.

### Main flow

1. create a session
2. open the SSE event stream
3. initialise audio
4. create a source
5. upload a file
6. start streaming
7. update positions
8. delete the session when finished

### Main endpoints

```text
POST   /api/session
GET    /api/<sid>/events
POST   /api/<sid>/init
POST   /api/<sid>/sources
POST   /api/<sid>/sources/<source_id>/upload
PATCH  /api/<sid>/sources/<source_id>/position
POST   /api/<sid>/sources/<source_id>/stream
DELETE /api/session/<sid>
```

Audio is returned through SSE as `audio_chunk` events.

## Browser Benchmark

The browser page includes a benchmark that runs the same end-to-end workflow for both protocols.

It measures:

- transport round-trip time
- time to first audio chunk
- chunk timing consistency
- approximate network latency
- memory used during initialisation
- position update round-trip time

This makes it easier to compare WebSocket and HTTP under the same client-side conditions.

## Running Tests

### Unit and integration tests

```bash
python -m pytest tests/ -v
```

### Load testing with Locust

```bash
locust -f tests/locustfile.py \
  --host http://<server>:5000 \
  --users 50 \
  --spawn-rate 5 \
  --run-time 60s \
  --csv results/n50
```

The load test creates sessions, initialises audio, uploads a test file, starts streaming, sends position updates, and records the timing results.

## Evaluation Summary

The project was tested locally and under distributed load. The main cloud-based evaluation used an AWS EC2 `c5a.2xlarge` instance with `8` vCPUs and `16 GB` RAM.

At a high level, the current results show:

- audio streaming remains stable at relatively high concurrency
- session setup and file upload are generally reliable under load
- position updates become the main bottleneck as concurrency increases
- Python runtime behaviour and library-level constraints limit update scalability more than streaming itself

This means the project supports the broader conclusion that server-side spatial audio is feasible, but its scalability depends strongly on where the processing bottlenecks sit.

## Main Libraries

| Library | What it is used for |
|---|---|
| `py3dti` | HRTF-based binaural rendering |
| `Flask` | server framework |
| `Flask-SocketIO` | WebSocket event handling |
| `gevent` | async I/O |
| `gevent-websocket` | WebSocket worker support |
| `gunicorn` | application server |
| `numpy` / `scipy` | audio processing support |
| `miniaudio` | audio decoding |
| `psutil` | memory measurement |
| `locust` | load testing |
