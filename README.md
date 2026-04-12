# SpatialSocket

![Music Happy](https://media.tenor.com/8nm7Ta8ZP74AAAAj/music-happy.gif)

Ever wanted to make audio feel like it's coming from different places in 3D space? That's exactly what SpatialSocket does! It's a real-time spatial audio rendering server I built for my final year project.

Here's the deal: SpatialSocket takes regular mono audio files, sprinkles some HRTF (head-related transfer function) magic on them using py3dti, and streams binaural audio to clients over WebSocket. I also threw in an HTTP+SSE version just for comparison.

Oh, and I didn't stop there - there's a browser benchmark, Locust load tests, and I even documented how it scales up to 100 concurrent sessions.

## How It Works

Every client that connects gets its very own audio processor with a personal HRTF renderer. You can place audio sources anywhere in 3D space and move them around in real-time - the server will keep rendering and streaming processed audio buffers like clockwork. 

Out of the box, I'm using the p0200 SOFA HRTF dataset at 48 kHz with 1024-sample buffers (that's 21.3 ms per render tick, if you're curious).

Here's another trick: I pre-load a bunch of processors in a renderer pool when the server starts up. This means new sessions get an initialised renderer instantly instead of waiting 8.7 seconds for HRTF loading every time someone connects.

## The Architecture

```
client (browser / locust)
        |
        | WebSocket (Socket.IO)       HTTP+SSE
        |                             |
   app/main.py                 http_server/main.py
   port 5000                   port 5001
        |
   gevent event loop
        |
   _global_render_loop (background thread)
        |
   _render_pool (GeventThreadPool, 50 threads)
        |
   py3dti AudioProcessor (one per session)
```

The WebSocket server runs as a single gunicorn worker with `GeventWebSocketWorker`. Everything happens in-process, which keeps things simple. The global render loop chugs along in the background, driving all active streaming sessions in real-time and passing DSP work to a thread pool. 

Here's the cool part: py3dti releases the GIL during rendering, so all that heavy DSP work gets parallelised across threads nicely.

## What You Need

- Python 3.9+
- libsndfile (`libsndfile1` on Debian/Ubuntu)
- Docker and Docker Compose (if you want to run it in containers)

Python dependencies are in `requirements.txt`. For development and load testing stuff, check out `requirements-dev.txt`.

## Running It Locally

```bash
# create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# install the good stuff
pip install -r requirements.txt

# start the WebSocket server
cd app
gunicorn \
  --worker-class geventwebsocket.gunicorn.workers.GeventWebSocketWorker \
  --workers 1 \
  --bind 0.0.0.0:5000 \
  --timeout 300 \
  main:app

# in another terminal, start the HTTP server
cd http_server
gunicorn \
  --worker-class gevent \
  --workers 1 \
  --bind 0.0.0.0:5001 \
  --timeout 300 \
  main:app
```

Open `index.html` in your browser and point it to `http://localhost:5000` and `http://localhost:5001`.

## Docker Way

```bash
docker compose up --build
```

Boom - WebSocket server on port 5000, HTTP server on port 5001. The renderer pool size is set to 100 by default in `docker-compose.yml`.

I also threw in cAdvisor, Prometheus and Grafana for keeping an eye on container resources during load tests (ports 8080, 9090 and 3000 respectively).

You can check how the pool filling is going anytime:

```bash
curl http://localhost:5000/health
# {"status":"healthy","renderer_pool_available":84,"renderer_pool_size":100}
```

The pool fills up one by one in the background (takes about 8.7 seconds per renderer). The server will take connections while it's still filling, but if you connect before it's ready, your session will just initialise its renderer on the spot.

## Tweaking Settings

You can override all these settings with environment variables:

| Variable | Default | What it does |
|---|---|---|
| `RENDERER_POOL_SIZE` | `0` | How many renderers to pre-load at startup |
| `BUFFER_SIZE` | `1024` | Samples per render tick (1024 = 21.3 ms at 48 kHz) |
| `SAMPLE_RATE` | `48000` | Audio sample rate in Hz |
| `MAX_SOURCES_PER_SESSION` | `16` | Max simultaneous sources per session |
| `SECRET_KEY` | none | Flask secret key (you'll need this in production) |
| `HRTF_DATASETS_DIR` | `./hrtf_datasets` | Where to find SOFA dataset files |

## WebSocket API

Everything uses Socket.IO. Connect to the server and follow along:

**Getting started**

| Event (emit) | Payload | Response event |
|---|---|---|
| `init_audio` | `{}` or `{hrtf_file: "p0200.sofa"}` | `status` with code `AUDIO_INITIALISED` |
| `create_source` | `{source_id, position: {x,y,z}}` | `status` with code `SOURCE_CREATED` |

**Audio upload and streaming**

First, upload your audio via HTTP POST:

```
POST /upload/<session_id>/<source_id>
Content-Type: multipart/form-data
field: audio_file
```

Then start the streaming:

| Event (emit) | Payload | Response event |
|---|---|---|
| `stream_uploaded_file` | `{source_id, filename}` | `file_audio_chunk` continuously |
| `stop_file_stream` | `{}` | `status` with code `STREAM_STOPPED` |

**Moving stuff around**

```json
emit('update_position', {
  "source_id": "src1",
  "position": {"x": 1.0, "y": 0.0, "z": 0.0}
})
```

Server hits you back with `status` code `POSITION_UPDATED`.

**What you get**

Each `file_audio_chunk` event contains:

```json
{
  "audio_data": "<base64 interleaved stereo PCM>",
  "source_positions": {"src1": {"current": 42, "total": 300}},
  "render_ts": 1743280000000
}
```

The `audio_data` is base64-encoded 32-bit float PCM, interleaved left/right channels at whatever sample rate you configured.

## HTTP+SSE API

If you're more of a REST person, create a session, subscribe to the event stream, then hit the REST endpoints for whatever you need to do.

```bash
# create session
POST /api/session
# returns {"session_id": "..."}

# subscribe to server events
GET /api/<sid>/events  (text/event-stream)

# initialise HRTF
POST /api/<sid>/init

# upload audio
POST /api/<sid>/sources/<source_id>/upload  (multipart, field: file)

# create source
POST /api/<sid>/sources  {"source_id": "src1", "position": {"x":1,"y":0,"z":0}}

# update position
PATCH /api/<sid>/sources/<source_id>/position  {"position": {"x":...,"y":...,"z":...}}

# start streaming
POST /api/<sid>/sources/<source_id>/stream  {"filename": "..."}

# delete session
DELETE /api/session/<sid>
```

Audio shows up as `audio_chunk` SSE events with the same payload format as the WebSocket version.

## Browser Benchmark

Open `index.html` and find the Performance Benchmark section. The benchmark runs a complete session (connect, HRTF init, upload, source creation, 10 position updates and streaming) for both WebSocket and HTTP+SSE, then tells you how it did:

| Metric | What it tells you |
|---|---|
| Transport RTT | How fast `create_source` travels (socketio event for WS, HTTP POST for SSE) |
| Pipeline latency | Time from hitting stream to getting your first audio chunk |
| Network latency | Rough one-way latency using server timestamp (good locally, not so much remotely) |
| Chunk jitter | How consistent the chunk arrival times are |
| Session memory | Memory used during HRTF init, measured on the server |
| Position update RTT | Average round-trip time for 10 position updates |

## Load Testing

Grab Locust from `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt
```

Run a test against your server:

```bash
locust -f tests/locustfile.py \
  --host http://<server>:5000 \
  --users 50 --spawn-rate 5 --run-time 60s \
  --csv results/n50
```

Each fake user connects, initialises audio, uploads a test tone, creates a source, streams for 3 seconds, does 10 position updates, then just hangs out. Results get saved to the `results/` folder.


## How It Scales

I ran these tests on an AWS EC2 c5a.2xlarge (8 vCPU AMD, 16 GB RAM) with 100 pre-loaded renderers, spawning 5 users per second.

| Users | Connect (median) | HRTF init (median) | First chunk (median) | Pos update (p95) | Failures |
|---|---|---|---|---|---|
| 10 | 110 ms | 25 ms | 35 ms | 57 ms | 0% |
| 25 | 300 ms | 92 ms | 120 ms | 1300 ms | 0% |
| 50 | 310 ms | 100 ms | 180 ms | 3000 ms | 0% |
| 100 | 540 ms | 190 ms | 300 ms | 5000 ms | 11% (pos update only) |

Connections, streaming and audio uploads work perfectly at 100% across the board. Position updates get a bit slow at scale because of GIL serialisation in py3dti's position update code. Check out `results/findings.md` for the full breakdown.

## What's Where

```
app/                    WebSocket server stuff
  main.py               entry point, socket handlers, renderer pool, render loop
  audio_processor.py    py3dti wrapper
  session_manager.py    session lifecycle
  source_manager.py     file handling and audio conversion
  config.py             configuration

http_server/            HTTP+SSE server
  main.py               REST API and SSE event stream

hrtf_datasets/          SOFA HRTF files
tests/
  locustfile.py         load test
  unit_tests.py
  integration_tests.py

results/                load test outputs and findings
  findings.md           detailed scaling analysis

index.html              demo interface and browser benchmark
docker-compose.yml
Dockerfile
prometheus.yml
```

## The Toolbox

| Library | What it does |
|---|---|
| py3dti | Binaural HRTF rendering |
| Flask + Flask-SocketIO | WebSocket and HTTP server |
| gevent + gevent-websocket | Async I/O and WebSocket support |
| gunicorn | Production WSGI server |
| numpy / scipy | Audio buffer processing |
| miniaudio | Audio decoding |
| psutil | Server-side memory measurement |
| locust | Load testing |
