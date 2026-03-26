"""
spatialsocket http api - http + sse version of the websocket api.
"""

from dotenv import load_dotenv
load_dotenv()

import os
import time
import json
import queue
import uuid
import logging
import threading

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.config import config
from app.audio_processor import AudioProcessor
from app.source_manager import SourceManager
from app.audio_streamer import encode_audio_to_base64

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.mp3', '.wav', '.m4a', '.ogg']
CORS(app, resources={r"/*": {"origins": "*"}})

HTTP_UPLOAD_DIR = os.environ.get('UPLOAD_DIR', os.path.join(os.path.dirname(__file__), 'uploads'))
os.makedirs(HTTP_UPLOAD_DIR, exist_ok=True)
os.makedirs(config.HRTF_UPLOAD_DIR, exist_ok=True)

_sessions:        dict = {}
_processors:      dict = {}
_source_managers: dict = {}
_event_queues:    dict = {}
_stream_states:   dict = {}
_stream_locks:    dict = {}
_stream_stops:    dict = {}
_stream_paused:   dict = {}
_buffer_cache:    dict = {}  # file_path -> pre-decoded buffers

def _push(sid: str, event: str, data: dict):
    """push event to session's sse queue."""
    q = _event_queues.get(sid)
    if q:
        try:
            q.put_nowait({'event': event, 'data': data})
        except queue.Full:
            logging.warning(f"SSE queue full for {sid}, dropping {event}")


def _sse_format(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.route('/api/session', methods=['POST'])
def create_session():
    sid = str(uuid.uuid4())
    _sessions[sid] = {'created_at': time.time()}
    _event_queues[sid] = queue.Queue(maxsize=400)
    _source_managers[sid] = SourceManager(upload_dir=HTTP_UPLOAD_DIR)
    logging.info(f"Session created: {sid}")
    return jsonify({
        'session_id': sid,
        'sample_rate': config.SAMPLE_RATE,
        'buffer_size': config.BUFFER_SIZE,
    })


@app.route('/api/session/<sid>', methods=['DELETE'])
def delete_session(sid):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    _stop_all(sid)
    proc = _processors.pop(sid, None)
    if proc:
        proc.cleanup()
    sm = _source_managers.pop(sid, None)
    if sm:
        sm.cleanup_session_files(sid)
    q = _event_queues.pop(sid, None)
    if q:
        try:
            q.put_nowait(None)  # sentinel → closes SSE generator
        except queue.Full:
            pass
    _sessions.pop(sid, None)
    logging.info(f"Session deleted: {sid}")
    return jsonify({'ok': True})


@app.route('/api/<sid>/events')
def event_stream(sid):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404

    def generate():
        q = _event_queues.get(sid)
        if not q:
            return
        yield ': connected\n\n'
        while True:
            try:
                item = q.get(timeout=30)
                if item is None:  # sentinel from delete_session
                    break
                yield _sse_format(item['event'], item['data'])
            except queue.Empty:
                yield ': heartbeat\n\n'  # keep connection alive

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',
        headers={
            'Cache-Control':    'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection':       'keep-alive',
        },
    )


@app.route('/api/<sid>/init', methods=['POST'])
def init_audio(sid):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404

    data = request.get_json(silent=True) or {}
    hrtf_file  = data.get('hrtf_file', config.DEFAULT_HRTF_FILE)
    use_custom = data.get('use_custom_hrtf', False)

    old = _processors.pop(sid, None)
    if old:
        old.cleanup()

    proc = AudioProcessor(
        sample_rate=config.SAMPLE_RATE,
        buffer_size=config.BUFFER_SIZE,
        max_sources=config.MAX_SOURCES_PER_SESSION,
    )
    _processors[sid] = proc
    _push(sid, 'status', {'code': 'AUDIO_LOADING', 'message': 'Loading HRTF…'})

    def _worker():
        import psutil
        _p = psutil.Process()
        rss_before = _p.memory_info().rss

        hrtf_path = None
        if use_custom:
            candidate = os.path.join(config.HRTF_UPLOAD_DIR, sid, secure_filename(hrtf_file))
            if os.path.exists(candidate):
                hrtf_path = candidate

        ok = proc.initialise(
            hrtf_file=hrtf_file if not hrtf_path else None,
            hrtf_path=hrtf_path,
        )
        rss_after = _p.memory_info().rss
        memory_delta_mb = round((rss_after - rss_before) / (1024 * 1024), 2)

        if ok and not proc.placeholder_mode:
            _push(sid, 'status', {'code': 'AUDIO_INITIALISED', 'message': 'Audio ready',
                                   'sample_rate': config.SAMPLE_RATE,
                                   'buffer_size':  config.BUFFER_SIZE,
                                   'memory_delta_mb': memory_delta_mb,
                                   'server_ts': int(time.time() * 1000)})
        else:
            _push(sid, 'status', {'code': 'INIT_FAILED',
                                   'error': 'HRTF load failed or placeholder mode',
                                   'server_ts': int(time.time() * 1000)})

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({'ok': True, 'message': 'Initialising…'})


@app.route('/api/<sid>/hrtf', methods=['POST'])
def upload_hrtf(sid):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if not f.filename or not f.filename.lower().endswith('.sofa'):
        return jsonify({'error': 'Must be a .sofa file'}), 400

    dest_dir = os.path.join(config.HRTF_UPLOAD_DIR, sid)
    os.makedirs(dest_dir, exist_ok=True)
    saved = secure_filename(f.filename)
    f.save(os.path.join(dest_dir, saved))
    return jsonify({'code': 'HRTF_UPLOADED', 'saved_filename': saved})


@app.route('/api/<sid>/sources', methods=['POST'])
def create_source(sid):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    data = request.get_json(silent=True) or {}
    source_id = data.get('source_id')
    position  = data.get('position', {'x': 0.0, 'y': 0.0, 'z': 0.0})

    if not source_id:
        return jsonify({'error': 'source_id required'}), 400

    proc = _processors.get(sid)
    if not proc or not proc.is_initialised:
        return jsonify({'code': 'CREATE_FAILED', 'source_id': source_id,
                        'message': 'Audio not initialised'}), 400

    ok = proc.create_source(source_id, position)
    if ok:
        return jsonify({'code': 'SOURCE_CREATED', 'source_id': source_id, 'position': position})
    return jsonify({'code': 'CREATE_FAILED', 'source_id': source_id,
                    'message': 'Duplicate source_id or limit reached'}), 400


@app.route('/api/<sid>/sources/<source_id>', methods=['DELETE'])
def remove_source(sid, source_id):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    proc = _processors.get(sid)
    if proc:
        proc.remove_source(source_id)
    return jsonify({'code': 'SOURCE_REMOVED', 'source_id': source_id})


@app.route('/api/<sid>/sources/<source_id>/position', methods=['PATCH'])
def update_position(sid, source_id):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    data     = request.get_json(silent=True) or {}
    position = data.get('position')
    if not position:
        return jsonify({'error': 'position required'}), 400
    proc = _processors.get(sid)
    if proc:
        proc.update_source_position(source_id, position)
    return jsonify({'code': 'POSITION_UPDATED', 'source_id': source_id, 'position': position,
                    'server_ts': int(time.time() * 1000)})


@app.route('/api/<sid>/sources/positions', methods=['PATCH'])
def batch_update_positions(sid):
    """update multiple source positions in one request."""
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    data    = request.get_json(silent=True) or {}
    updates = data.get('updates', [])
    proc    = _processors.get(sid)
    results = []
    for u in updates:
        src_id = u.get('source_id')
        pos    = u.get('position')
        if src_id and pos and proc:
            ok = proc.update_source_position(src_id, pos)
            results.append({'source_id': src_id, 'ok': ok})
        else:
            results.append({'source_id': src_id, 'ok': False, 'error': 'missing fields or no processor'})
    return jsonify({'code': 'BATCH_POSITIONS_UPDATED', 'results': results,
                    'server_ts': int(time.time() * 1000)})


@app.route('/api/<sid>/listener', methods=['PATCH'])
def update_listener(sid):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    data        = request.get_json(silent=True) or {}
    position    = data.get('position')
    orientation = data.get('orientation')
    proc = _processors.get(sid)
    if proc and (position or orientation):
        proc.set_listener_pose(
            position    or proc.listener_pose['position'],
            orientation or proc.listener_pose['orientation'],
        )
    return jsonify({'code': 'LISTENER_UPDATED'})


@app.route('/api/<sid>/sources/<source_id>/upload', methods=['POST'])
def upload_audio(sid, source_id):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in app.config['UPLOAD_EXTENSIONS']:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    sm = _source_managers.get(sid)
    if not sm:
        return jsonify({'error': 'Session not found'}), 404

    path = sm.save_uploaded_file_stream(sid, source_id, f, f.filename)
    if not path:
        return jsonify({'error': 'Failed to save file'}), 500

    buffers = sm.convert_mp3_to_audio_buffers(path, config.SAMPLE_RATE, config.BUFFER_SIZE)
    if buffers:
        _buffer_cache[path] = buffers

    info = sm.get_audio_info(path)
    return jsonify({
        'code':             'UPLOAD_OK',
        'source_id':        source_id,
        'saved_filename':   os.path.basename(path),
        'duration_seconds': info.get('duration_seconds', 0) if info else 0,
        'sample_rate':      info.get('sample_rate', 0)      if info else 0,
    })


@app.route('/api/<sid>/sources/<source_id>/stream', methods=['POST'])
def start_stream(sid, source_id):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    data     = request.get_json(silent=True) or {}
    filename = data.get('filename')
    if not filename:
        return jsonify({'error': 'filename required'}), 400

    proc = _processors.get(sid)
    if not proc or not proc.is_initialised:
        return jsonify({'error': 'Audio not initialised'}), 400

    sm = _source_managers.get(sid)
    if not sm:
        return jsonify({'error': 'Session not found'}), 404

    session_dir = os.path.join(sm.upload_dir, sid)
    file_path   = os.path.join(session_dir, secure_filename(filename))
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    _push(sid, 'status', {'code': 'DECODING', 'source_id': source_id,
                           'message': 'Decoding audio…', 'server_ts': int(time.time() * 1000)})

    def _decode():
        buffers = _buffer_cache.get(file_path) or sm.convert_mp3_to_audio_buffers(file_path, config.SAMPLE_RATE, config.BUFFER_SIZE)
        if not buffers:
            _push(sid, 'status', {'code': 'DECODE_FAILED', 'source_id': source_id,
                                   'message': 'Failed to decode audio'})
            return

        if sid not in _stream_states:
            _stream_states[sid] = {}
            _stream_locks[sid]  = threading.Lock()

        with _stream_locks[sid]:
            _stream_states[sid][source_id] = {'buffers': buffers, 'pos': 0}

        dur = len(buffers) * config.BUFFER_SIZE / config.SAMPLE_RATE
        _push(sid, 'status', {
            'code':            'STREAM_READY',
            'source_id':       source_id,
            'total_buffers':   len(buffers),
            'duration_seconds': dur,
            'server_ts':       int(time.time() * 1000),
        })

        if sid not in _stream_stops:
            stop_ev = threading.Event()
            _stream_stops[sid] = stop_ev
            threading.Thread(target=_stream_task, args=(sid, stop_ev), daemon=True).start()

    threading.Thread(target=_decode, daemon=True).start()
    return jsonify({'ok': True, 'message': 'Decoding…'})


@app.route('/api/<sid>/sources/<source_id>/stream/pause', methods=['POST'])
def pause_stream(sid, source_id):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    _stream_paused.setdefault(sid, set()).add(source_id)
    _push(sid, 'status', {'code': 'STREAM_PAUSED', 'source_id': source_id})
    return jsonify({'code': 'STREAM_PAUSED', 'source_id': source_id})


@app.route('/api/<sid>/sources/<source_id>/stream/resume', methods=['POST'])
def resume_stream(sid, source_id):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    if sid in _stream_paused:
        _stream_paused[sid].discard(source_id)
    _push(sid, 'status', {'code': 'STREAM_RESUMED', 'source_id': source_id})
    return jsonify({'code': 'STREAM_RESUMED', 'source_id': source_id})


@app.route('/api/<sid>/sources/<source_id>/stream/stop', methods=['POST'])
def stop_source(sid, source_id):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    lock = _stream_locks.get(sid)
    if lock:
        with lock:
            _stream_states.get(sid, {}).pop(source_id, None)
    if sid in _stream_paused:
        _stream_paused[sid].discard(source_id)
    _push(sid, 'status', {'code': 'SOURCE_STOPPED', 'source_id': source_id})
    return jsonify({'code': 'SOURCE_STOPPED', 'source_id': source_id})


@app.route('/api/<sid>/stream/stop', methods=['POST'])
def stop_all_streams(sid):
    if sid not in _sessions:
        return jsonify({'error': 'Session not found'}), 404
    _stop_all(sid)
    _push(sid, 'status', {'code': 'STREAM_STOPPED'})
    return jsonify({'code': 'STREAM_STOPPED'})


def _stop_all(sid: str):
    ev = _stream_stops.pop(sid, None)
    if ev:
        ev.set()
    _stream_states.pop(sid, None)
    _stream_locks.pop(sid, None)
    _stream_paused.pop(sid, None)


def _stream_task(sid: str, stop_event: threading.Event):
    buffer_duration = config.BUFFER_SIZE / config.SAMPLE_RATE
    t_start = time.time()
    tick    = 0

    while not stop_event.is_set():
        lock  = _stream_locks.get(sid)
        state = _stream_states.get(sid)
        if lock is None or state is None:
            time.sleep(0.01)
            continue

        with lock:
            if not state:
                break
            buffers_map      = {}
            source_positions = {}
            finished         = []
            paused           = _stream_paused.get(sid, set())

            for src_id, src in state.items():
                if src_id in paused:
                    continue
                pos   = src['pos']
                total = len(src['buffers'])
                if pos >= total:
                    finished.append(src_id)
                    continue
                buffers_map[src_id]      = src['buffers'][pos]
                src['pos']               = pos + 1
                source_positions[src_id] = {'current': pos, 'total': total}

            for src_id in finished:
                del state[src_id]

        for src_id in finished:
            _push(sid, 'status', {'code': 'TRACK_COMPLETE', 'source_id': src_id,
                                   'server_ts': int(time.time() * 1000)})

        if not buffers_map:
            with lock:
                if not state:
                    break
            tick += 1
            continue

        proc = _processors.get(sid)
        if not proc or not proc.is_initialised:
            return

        result = proc.process_sources(buffers_map)
        if result is not None:
            left, right = result
            _push(sid, 'audio_chunk', {
                'audio_data':       encode_audio_to_base64(left, right),
                'source_positions': source_positions,
                'render_ts':        int(time.time() * 1000),
            })

        tick  += 1
        t_next = t_start + tick * buffer_duration
        sleep  = t_next - time.time()
        if sleep > 0:
            time.sleep(sleep)
        elif sleep < -10 * buffer_duration:
            t_start = time.time() - tick * buffer_duration

    if not stop_event.is_set():
        _push(sid, 'status', {'code': 'STREAM_COMPLETE', 'server_ts': int(time.time() * 1000)})
    _stream_stops.pop(sid, None)
    _stream_paused.pop(sid, None)
    logging.info(f"Stream task ended: {sid}")


@app.route('/')
def index():
    return jsonify({
        'service': 'SpatialSocket HTTP API',
        'version': '1.0.0',
        'status':  'running',
        'transport': 'HTTP+SSE',
    })


if __name__ == '__main__':
    port = int(os.environ.get('HTTP_PORT', 5001))
    print(f"Starting SpatialSocket HTTP API on port {port}…")
    app.run(host=config.HOST, port=port, debug=config.DEBUG, threaded=True)
