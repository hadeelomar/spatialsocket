from gevent import monkey
monkey.patch_all()

from dotenv import load_dotenv
load_dotenv()
import os
import time
import logging
import atexit
import threading
import gevent
from gevent.threadpool import ThreadPool as _GeventThreadPool
from flask import Flask, jsonify, request, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.utils import secure_filename
from typing import Dict
from config import config
from session_manager import SessionManager, StreamState
from audio_processor import AudioProcessor
from audio_streamer import (
    encode_audio_to_base64,
    decode_audio_from_base64,
    generate_test_tone
)
from performance_monitor import PerformanceMonitor
from source_manager import SourceManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.mp3', '.wav', '.m4a', '.ogg']
app.config['HRTF_EXTENSIONS'] = ['.sofa']
CORS(app, resources={r"/*": {"origins": config.CORS_ALLOWED_ORIGINS}})

os.makedirs(config.HRTF_UPLOAD_DIR, exist_ok=True)
RENDERED_DIR = os.path.join(os.getcwd(), 'rendered')
os.makedirs(RENDERED_DIR, exist_ok=True)


def allowed_file(filename: str, extensions: list = None) -> bool:
    if extensions is None:
        extensions = app.config['UPLOAD_EXTENSIONS']
    return '.' in filename and os.path.splitext(filename)[1].lower() in extensions

socketio = SocketIO(
    app,
    cors_allowed_origins=config.CORS_ALLOWED_ORIGINS,
    async_mode='gevent',
    ping_interval=25,
    ping_timeout=120,
)

session_manager = SessionManager(
    session_timeout=config.SESSION_TIMEOUT_SECONDS,
    cleanup_interval=config.SESSION_CLEANUP_INTERVAL
)
performance_monitor = PerformanceMonitor()
audio_processors: Dict[str, AudioProcessor] = {}
source_manager = SourceManager()
_g_stream_states: Dict[str, dict] = {}   # {session_id: {source_id: {'buffers': list, 'pos': int}}}
_g_stream_locks: Dict[str, threading.Lock] = {}
_g_stream_active: set = set()            # sessions the global render loop should tick
_g_stream_paused: Dict[str, set] = {}   # {session_id: set of paused source_ids}
_g_lock = threading.Lock()              # guards _g_stream_active
decoded_buffer_cache: Dict[str, list] = {}

_cpu = os.cpu_count() or 4
# py3dti releases GIL during DSP — real OS threads give true parallelism
# hrtf_pool=50: 100 sessions / 50 threads = 2 batches × ~5s = ~10s (within benchmark timeout)
# render_pool=50: covers all 100 sessions; GIL released so they run in parallel
# setup_pool=16: for interactive create_source / update_pos (20-50ms each)
_hrtf_pool   = _GeventThreadPool(50)   # parallel HRTF loading — dominant bottleneck for 100 sessions
_render_pool = _GeventThreadPool(50)   # parallel render ticks
_setup_pool  = _GeventThreadPool(16)   # interactive py3dti ops

def shutdown_server():
    logging.info("shutting down server...")
    for session_id, processor in list(audio_processors.items()):
        try:
            processor.cleanup()
            logging.info(f"cleaned up audio processor for session {session_id}")
        except Exception as e:
            logging.error(f"error cleaning up processor for {session_id}: {e}")
    audio_processors.clear()
    session_manager.shutdown()
    logging.info("server shutdown complete")

atexit.register(shutdown_server)

@app.route('/')
def index():
    return jsonify({
        'service': 'SpatialSocket API',
        'version': '0.4.0',
        'status': 'running',
        'active_sessions': session_manager.get_session_count(),
        'performance_metrics': performance_monitor.to_dict()['global']
    })

@app.route('/metrics')
def get_metrics():
    return jsonify(performance_monitor.to_dict())

@app.route('/sessions')
def get_sessions():
    return jsonify({
        'sessions': session_manager.get_all_session_info(),
        'total_sessions': session_manager.get_session_count()
    })

@app.route('/rendered/<session_id>/<filename>')
def serve_rendered_audio(session_id, filename):
    try:
        file_path = os.path.join(RENDERED_DIR, session_id, secure_filename(filename))
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(file_path, mimetype='audio/wav')
    except Exception as e:
        logging.error(f"Error serving rendered file: {e}")
        return jsonify({'error': 'Failed to serve file'}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/upload/<session_id>/<source_id>', methods=['POST'])
def upload_audio_file(session_id, source_id):
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Allowed: MP3, WAV, M4A, OGG'}), 400
        
        filename = file.filename
        file_path = source_manager.save_uploaded_file_stream(session_id, source_id, file, filename)

        if not file_path:
            return jsonify({'error': 'Failed to save file'}), 500

        saved_filename = os.path.basename(file_path)
        audio_info = source_manager.get_audio_info(file_path)

        buffers = source_manager.convert_mp3_to_audio_buffers(file_path, config.SAMPLE_RATE, config.BUFFER_SIZE)
        if buffers:
            decoded_buffer_cache[file_path] = buffers
            logging.info(f"pre-decoded {saved_filename}: {len(buffers)} buffers")

        return jsonify({
            'message': 'File uploaded successfully',
            'file_info': audio_info,
            'saved_filename': saved_filename,
            'source_id': source_id,
            'session_id': session_id
        })
        
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/upload_hrtf/<session_id>', methods=['POST'])
def upload_hrtf_file(session_id):
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404

        if 'hrtf_file' not in request.files:
            return jsonify({'error': 'No hrtf_file field in request'}), 400

        file = request.files['hrtf_file']
        if not file.filename:
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename, app.config['HRTF_EXTENSIONS']):
            return jsonify({'error': 'Only .sofa files are accepted for HRTF upload'}), 400

        session_hrtf_dir = os.path.join(config.HRTF_UPLOAD_DIR, session_id)
        os.makedirs(session_hrtf_dir, exist_ok=True)

        safe_name = secure_filename(file.filename)
        save_path = os.path.join(session_hrtf_dir, safe_name)
        file.save(save_path)

        logging.info(f"HRTF uploaded for session {session_id}: {save_path}")
        return jsonify({
            'message': 'HRTF uploaded successfully',
            'filename': safe_name,
            'session_id': session_id,
            'instruction': 'Emit init_audio with use_uploaded=true to activate this HRTF'
        })

    except Exception as e:
        logging.error(f"HRTF upload error: {e}")
        return jsonify({'error': f'HRTF upload failed: {str(e)}'}), 500


@app.route('/files/<session_id>', methods=['GET'])
def list_session_files(session_id):
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        files = source_manager.get_session_files(session_id)
        return jsonify({
            'session_id': session_id,
            'files': files,
            'total_files': len(files)
        })
        
    except Exception as e:
        logging.error(f"List files error: {e}")
        return jsonify({'error': f'Failed to list files: {str(e)}'}), 500

@app.route('/stream/<session_id>/<source_id>/<filename>', methods=['GET'])
def stream_audio_file(session_id, source_id, filename):
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        session_dir = os.path.join(source_manager.upload_dir, session_id)
        file_path = os.path.join(session_dir, secure_filename(filename))
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        buffers = source_manager.convert_mp3_to_audio_buffers(
            file_path, 
            config.SAMPLE_RATE, 
            config.BUFFER_SIZE
        )
        
        if not buffers:
            return jsonify({'error': 'Failed to process audio file'}), 500
        
        return jsonify({
            'message': 'Audio processed for streaming',
            'source_id': source_id,
            'filename': filename,
            'total_buffers': len(buffers),
            'buffer_size': config.BUFFER_SIZE,
            'sample_rate': config.SAMPLE_RATE,
            'duration_seconds': len(buffers) * config.BUFFER_SIZE / config.SAMPLE_RATE
        })
        
    except Exception as e:
        logging.error(f"Stream preparation error: {e}")
        return jsonify({'error': f'Failed to prepare stream: {str(e)}'}), 500

@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    logging.info(f"Client connected: {session_id}")

    performance_monitor.record_receive_timestamp(session_id)

    recovery_token = (request.args or {}).get('recovery_token')
    recovered = False
    session = None

    if recovery_token:
        old_session = session_manager.find_by_recovery_token(recovery_token)
        if old_session:
            old_id = old_session.session_id
            session = session_manager.migrate_session(old_id, session_id)
            if session and old_id in audio_processors:
                audio_processors[session_id] = audio_processors.pop(old_id)
                if old_id in _g_stream_states:
                    _g_stream_states[session_id] = _g_stream_states.pop(old_id)
                    _g_stream_locks[session_id] = _g_stream_locks.pop(old_id, threading.Lock())
                    with _g_lock:
                        if old_id in _g_stream_active:
                            _g_stream_active.discard(old_id)
                            _g_stream_active.add(session_id)
            recovered = True

    if not recovered:
        session = session_manager.create_session(session_id, config.SAMPLE_RATE, config.BUFFER_SIZE)
        audio_processors[session_id] = AudioProcessor(config.SAMPLE_RATE, config.BUFFER_SIZE)
        session.set_stream_state(StreamState.IDLE)

    performance_monitor.record_send_timestamp(session_id)
    
    emit('connected', {
        'session_id': session_id,
        'sample_rate': config.SAMPLE_RATE,
        'buffer_size': config.BUFFER_SIZE,
        'message': 'Connected to SpatialSocket API'
    })
    
    logging.info(f"session {session_id} fully initialised")

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    logging.info(f"Client disconnected: {session_id}")
    performance_monitor.record_receive_timestamp(session_id)
    
    session = session_manager.get_session(session_id)
    if session:
        session.set_stream_state(StreamState.STOPPING)
    
    _stop_file_stream(session_id)

    if session_id in audio_processors:
        try:
            processor = audio_processors[session_id]
            processor.cleanup()
            del audio_processors[session_id]
            logging.info(f"audio processor cleaned up for session {session_id}")
        except Exception as e:
            logging.error(f"error cleaning up processor for {session_id}: {e}")
    
    session_manager.remove_session(session_id)
    performance_monitor.cleanup_session(session_id)
    
    try:
        source_manager.cleanup_session_files(session_id)
        logging.info(f"uploaded audio files cleaned up for session {session_id}")
    except Exception as e:
        logging.error(f"error cleaning up audio files for {session_id}: {e}")

    try:
        session_hrtf_dir = os.path.join(config.HRTF_UPLOAD_DIR, session_id)
        if os.path.isdir(session_hrtf_dir):
            for f in os.listdir(session_hrtf_dir):
                os.remove(os.path.join(session_hrtf_dir, f))
            os.rmdir(session_hrtf_dir)
            logging.info(f"uploaded hrtf files cleaned up for session {session_id}")
    except Exception as e:
        logging.error(f"error cleaning up hrtf files for {session_id}: {e}")
    
    try:
        session_rendered_dir = os.path.join(RENDERED_DIR, session_id)
        if os.path.isdir(session_rendered_dir):
            for f in os.listdir(session_rendered_dir):
                os.remove(os.path.join(session_rendered_dir, f))
            os.rmdir(session_rendered_dir)
            logging.info(f"rendered audio files cleaned up for session {session_id}")
    except Exception as e:
        logging.error(f"error cleaning up rendered files for {session_id}: {e}")
    
    logging.info(f"session {session_id} fully disconnected and cleaned up")

@socketio.on('init_audio')
def handle_init_audio(data=None):
    """initialise the audio processor in a background thread to avoid blocking socketio events."""
    session_id = request.sid
    processor = audio_processors.get(session_id)

    if not processor:
        emit('error', {'message': 'No audio processor found'})
        return

    data = data or {}
    use_uploaded = data.get('use_uploaded', False)

    hrtf_path = None
    hrtf_file = None

    if use_uploaded:
        session_hrtf_dir = os.path.join(config.HRTF_UPLOAD_DIR, session_id)
        if os.path.isdir(session_hrtf_dir):
            sofa_files = [f for f in os.listdir(session_hrtf_dir) if f.endswith('.sofa')]
            if sofa_files:
                sofa_files.sort(key=lambda f: os.path.getmtime(
                    os.path.join(session_hrtf_dir, f)), reverse=True)
                hrtf_path = os.path.join(session_hrtf_dir, sofa_files[0])
                logging.info(f"session {session_id}: using uploaded hrtf {hrtf_path}")
            else:
                emit('error', {'code': 'NO_UPLOADED_HRTF',
                               'message': 'use_uploaded=true but no SOFA file uploaded'})
                return
    else:
        hrtf_file = data.get('hrtf_file', config.DEFAULT_HRTF_FILE)

    emit('status', {
        'code': 'AUDIO_LOADING',
        'message': f'Loading HRTF {"(uploaded)" if hrtf_path else hrtf_file}…',
        'server_ts': int(time.time() * 1000),
    })

    def _init_worker():
        import psutil
        _proc = psutil.Process()
        rss_before = _proc.memory_info().rss
        try:
            # run py3dti in a real OS thread via _hrtf_pool so the gevent event loop
            # stays responsive while the HRTF file is loading (py3dti holds the GIL)
            success = _hrtf_pool.apply(
                processor.initialise,
                kwds={'hrtf_file': hrtf_file, 'hrtf_path': hrtf_path},
            )
        except BaseException as e:
            logging.error(f"session {session_id}: _init_worker crashed: {e}")
            success = False
        rss_after = _proc.memory_info().rss
        memory_delta_mb = round((rss_after - rss_before) / (1024 * 1024), 2)
        logging.info(f"session {session_id}: emitting {'audio_initialised' if success else 'init_failed'} (mem_delta={memory_delta_mb:.1f}MB)")
        socketio.emit('status', {
            'code': 'AUDIO_INITIALISED' if success else 'INIT_FAILED',
            'message': 'audio initialised' if success else 'failed to initialise',
            'hrtf_source': hrtf_path if hrtf_path else hrtf_file,
            'placeholder_mode': processor.placeholder_mode,
            'memory_delta_mb': memory_delta_mb,
            'server_ts': int(time.time() * 1000),
        }, to=session_id, namespace='/')

    socketio.start_background_task(_init_worker)

@socketio.on('remove_source')
def handle_remove_source(data):
    session_id = request.sid
    processor = audio_processors.get(session_id)
    
    source_id = data.get('source_id')
    
    if not source_id:
        emit('error', {'message': 'source_id required'})
        return
    
    if processor:
        success = processor.remove_source(source_id)
        if success:
            performance_monitor.decrement_source_count(session_id)
        
        emit('status', {
            'code': 'SOURCE_REMOVED' if success else 'REMOVE_FAILED',
            'source_id': source_id,
            'remaining_sources': processor.get_source_count(),
            'server_ts': int(time.time() * 1000),
        })
    else:
        emit('error', {'message': 'No audio processor found'})

@socketio.on('create_source')
def handle_create_source(data):
    session_id = request.sid
    processor = audio_processors.get(session_id)

    source_id = data.get('source_id')
    position = data.get('position', {'x': 0, 'y': 0, 'z': 0})

    if not source_id:
        emit('error', {'message': 'source_id required'})
        return

    if processor:
        success = _setup_pool.apply(processor.create_source, (source_id, position))
        if success:
            performance_monitor.increment_source_count(session_id)
        
        emit('status', {
            'code': 'SOURCE_CREATED' if success else 'CREATE_FAILED',
            'message': f'source "{source_id}" created' if success else f'failed to create source "{source_id}" - is audio initialised?',
            'source_id': source_id,
            'position': position,
            'total_sources': processor.get_source_count(),
            'server_ts': int(time.time() * 1000),
        })
    else:
        emit('error', {'message': 'No audio processor found'})

@socketio.on('update_position')
def handle_update_position(data):
    session_id = request.sid
    processor = audio_processors.get(session_id)

    source_id = data.get('source_id')
    position = data.get('position')

    if not source_id or not position:
        emit('error', {'message': 'source_id and position required'})
        return

    if processor:
        success = _setup_pool.apply(processor.update_source_position, (source_id, position))
        resp = {
            'code': 'POSITION_UPDATED' if success else 'UPDATE_FAILED',
            'source_id': source_id,
            'server_ts': int(time.time() * 1000),
        }
        if '_seq' in data:
            resp['_seq'] = data['_seq']
        emit('status', resp)


@socketio.on('update_listener')
def handle_update_listener(data):
    """handle listener pose updates with validation and optimisation."""
    session_id = request.sid
    processor = audio_processors.get(session_id)
    session = session_manager.get_session(session_id)
    
    performance_monitor.record_receive_timestamp(session_id)
    performance_monitor.record_processing_start(session_id)
    
    validation_error = _validate_listener_payload(data)
    if validation_error:
        performance_monitor.increment_errors()
        emit('error', {
            'code': 'INVALID_PAYLOAD',
            'message': validation_error
        })
        return

    position = data['position']
    orientation = data['orientation']

    if not session:
        performance_monitor.increment_errors()
        emit('error', {
            'code': 'SESSION_NOT_FOUND',
            'message': 'Session not found'
        })
        return

    position_changed = session.update_listener_pose(position, orientation)

    if processor and processor.is_initialised:
        success = processor.set_listener_pose(position, orientation, position_changed)
        if not success:
            performance_monitor.increment_errors()
            emit('error', {
                'code': 'LISTENER_UPDATE_FAILED',
                'message': 'Failed to update listener pose in audio processor'
            })
            return
    
    performance_monitor.record_processing_end(session_id)
    performance_monitor.record_send_timestamp(session_id)
    
    emit('status', {
        'code': 'LISTENER_UPDATED',
        'message': 'Listener pose updated successfully',
        'position_changed': position_changed,
        'pose': session.get_listener_pose(),
        'server_ts': int(time.time() * 1000),
    })


def _validate_listener_payload(data: dict) -> str:
    """validate listener update payload; returns error string or none."""
    required_keys = ['position', 'orientation']
    for key in required_keys:
        if key not in data:
            return f'missing required key: {key}'
    
    position = data['position']
    orientation = data['orientation']

    if not isinstance(position, dict):
        return 'position must be a dictionary'
    
    pos_required_keys = ['x', 'y', 'z']
    for key in pos_required_keys:
        if key not in position:
            return f'missing required position key: {key}'
        if not isinstance(position[key], (int, float)):
            return f'position {key} must be a number'
        if abs(position[key]) > 1000:
            return f'position {key} value {position[key]} is outside reasonable range (-1000 to 1000)'

    if not isinstance(orientation, dict):
        return 'orientation must be a dictionary'
    
    orient_required_keys = ['forward', 'up']
    for key in orient_required_keys:
        if key not in orientation:
            return f'missing required orientation key: {key}'
        if not isinstance(orientation[key], dict):
            return f'orientation {key} must be a dictionary'

        vector = orientation[key]
        vector_required_keys = ['x', 'y', 'z']
        for vkey in vector_required_keys:
            if vkey not in vector:
                return f'missing required orientation {key} key: {vkey}'
            if not isinstance(vector[vkey], (int, float)):
                return f'orientation {key} {vkey} must be a number'
            if abs(vector[vkey]) > 1.1:  # small tolerance for floating point
                return f'orientation {key} {vkey} value {vector[vkey]} is outside unit vector range (-1 to 1)'

    forward = orientation['forward']
    up = orientation['up']
    dot_product = (forward['x'] * up['x'] + forward['y'] * up['y'] + forward['z'] * up['z'])
    if abs(dot_product) > 0.99:  # nearly parallel
        return 'forward and up vectors should not be parallel'

    return None


@socketio.on('stream_audio')
def handle_stream_audio(data):
    session_id = request.sid
    processor = audio_processors.get(session_id)
    session = session_manager.get_session(session_id)
    
    performance_monitor.record_receive_timestamp(session_id)
    performance_monitor.record_processing_start(session_id)
    
    if session:
        session.set_stream_state(StreamState.STREAMING)

    if not processor or not processor.is_initialised:
        performance_monitor.increment_errors()
        if session:
            session.set_stream_state(StreamState.ERROR)
        emit('error', {
            'code': 'PROCESSOR_NOT_READY',
            'message': 'Audio processor not initalised'
        })
        return
    
    source_id = data.get('source_id')
    audio_data_encoded = data.get('audio_data')

    if not source_id or not audio_data_encoded:
        performance_monitor.increment_errors()
        if session:
            session.set_stream_state(StreamState.ERROR)
        emit('error', {
            'code': 'INVALID_DATA',
            'message': 'source_id and audio_data required'
        })
        return

    audio_buffer = decode_audio_from_base64(audio_data_encoded, config.BUFFER_SIZE)
    
    if audio_buffer is not None:
        performance_monitor.increment_bytes_in(session_id, len(audio_data_encoded))

    if audio_buffer is None:
        performance_monitor.increment_errors()
        if session:
            session.set_stream_state(StreamState.ERROR)
        emit('error', {
            'code': 'DECODE_ERROR',
            'message': 'Failed to decode audio data'
        })
        return
    
    result = _setup_pool.apply(processor.process_audio, (source_id, audio_buffer))
    if result is None:
        performance_monitor.increment_errors()
        if session:
            session.set_stream_state(StreamState.ERROR)
        emit('error', {
            'code': 'PROCESSING_ERROR',
            'message': 'Failed to process audio'
        })
        return

    left_channel, right_channel = result
    output_encoded = encode_audio_to_base64(left_channel, right_channel)
    
    performance_monitor.increment_frames_processed(session_id)
    performance_monitor.increment_bytes_out(session_id, len(output_encoded))
    performance_monitor.record_processing_end(session_id)
    performance_monitor.record_send_timestamp(session_id)
    
    if session:
        session.set_stream_state(StreamState.IDLE)
    
    emit('processed_audio', {
        'source_id': source_id,
        'audio_data': output_encoded,
        'timestamp': time.time()
    })

@socketio.on('request_test_tone')
def handle_request_test_tone(data):
    session_id = request.sid
    
    performance_monitor.record_receive_timestamp(session_id)
    performance_monitor.record_processing_start(session_id)
    
    print(f"[request_test_tone] Received request from session {session_id}")
    
    try:
        if session_id not in audio_processors:
            print(f"[request_test_tone] ERROR: No processor found for session {session_id}")
            performance_monitor.increment_errors()
            emit('error', {
                'code': 'SESSION_NOT_FOUND',
                'message': 'No audio processor found for session'
            })
            return
        
        processor = audio_processors[session_id]
        
        if not processor or not processor.is_initialised:
            print(f"[request_test_tone] ERROR: Processor not ready for session {session_id}")
            performance_monitor.increment_errors()
            emit('error', {
                'code': 'PROCESSOR_NOT_READY',
                'message': 'Audio processor not initialised'
            })
            return
        
        source_id = data.get('source_id', 'test_source')
        frequency = data.get('frequency', 440)
        duration = data.get('duration', 1.0)
        
        print(f"[request_test_tone] Processing test tone: source_id={source_id}, freq={frequency}, duration={duration}")
        
        if source_id not in processor.sources:
            if not processor.create_source(source_id, {'x': 0, 'y': 0, 'z': 0}):
                print(f"[request_test_tone] ERROR: Failed to create source {source_id}")
                emit('error', {
                    'code': 'SOURCE_CREATION_FAILED',
                    'message': f'Failed to create source {source_id}'
                })
                return
            print(f"[request_test_tone] Created new source {source_id}")
        else:
            print(f"[request_test_tone] Using existing source {source_id}")
        
        print(f"[request_test_tone] Source {source_id} created/verified")
        
        test_audio = generate_test_tone(frequency, duration, config.SAMPLE_RATE)
        buffer_size = config.BUFFER_SIZE
        num_chunks = len(test_audio) // buffer_size
        
        print(f"[request_test_tone] Processing {num_chunks} chunks")
        
        for i in range(num_chunks):
            start = i*buffer_size
            end = start + buffer_size
            chunk = test_audio[start:end]

            result = _setup_pool.apply(processor.process_audio, (source_id, chunk))

            if result:
                left_channel, right_channel = result
                output_encoded = encode_audio_to_base64(left_channel, right_channel)
                
                performance_monitor.increment_frames_processed(session_id)
                performance_monitor.increment_bytes_out(session_id, len(output_encoded))
                performance_monitor.record_send_timestamp(session_id)
                
                emit('processed_audio', {
                    'source_id': source_id,
                    'audio_data': output_encoded,
                    'chunk_index': i,
                    'total_chunks': num_chunks,
                    'timestamp': time.time()
                })
                print(f"[request_test_tone] Sent chunk {i+1}/{num_chunks}")
            else:
                print(f"[request_test_tone] ERROR: Failed to process chunk {i}")
                performance_monitor.increment_errors()
                emit('error', {
                    'code': 'PROCESSING_ERROR',
                    'message': f'Failed to process audio chunk {i}'
                })
                return
        
        performance_monitor.record_processing_end(session_id)
        emit('status', {
            'code': 'TEST_TONE_COMPLETE',
            'message': f'Processed {num_chunks} chunks',
            'source_id': source_id
        })
        print(f"[request_test_tone] Completed test tone for session {session_id}")
        
    except Exception as e:
        print(f"[request_test_tone] ERROR: Exception in handler: {str(e)}")
        performance_monitor.increment_errors()
        emit('error', {
            'code': 'HANDLER_ERROR',
            'message': f'Error processing test tone: {str(e)}'
        })

@socketio.on('set_data')
def handle_set_data(data):
    session_id = request.sid
    performance_monitor.record_receive_timestamp(session_id)
    session = session_manager.get_session(session_id)

    if session:
        session.data.update(data)
        session.touch()
        performance_monitor.record_send_timestamp(session_id)
        emit('status', {
            'message': 'Data stored',
            'data': session.data
        })
    else:
        performance_monitor.increment_errors()
        emit('error', {
            'message': 'Session not found'
        })

@socketio.on('get_data')
def handle_get_data():
    try:
        session_id = request.sid
        performance_monitor.record_receive_timestamp(session_id)
        session = session_manager.get_session(session_id)

        if session:
            session.touch()
            performance_monitor.record_send_timestamp(session_id)
            emit('status', {'message': 'Data retrieved', 'data': session.data})
        else:
            emit('error', {'message': 'No audio frames were rendered'})

    except Exception as e:
        logging.error(f"File rendering error: {e}")
        emit('error', {'message': f'File rendering failed: {str(e)}'})

def _stop_file_stream(session_id: str):
    """remove session from global render loop without emitting STREAM_COMPLETE."""
    with _g_lock:
        _g_stream_active.discard(session_id)
    _g_stream_states.pop(session_id, None)
    _g_stream_locks.pop(session_id, None)
    _g_stream_paused.pop(session_id, None)


@socketio.on('stream_uploaded_file')
def handle_stream_uploaded_file(data):
    """start streaming a previously uploaded file through spatial processing."""
    session_id = request.sid
    processor = audio_processors.get(session_id)
    session = session_manager.get_session(session_id)

    if not session:
        emit('error', {'message': 'Session not found'})
        return

    if not processor or not processor.is_initialised:
        emit('error', {'code': 'PROCESSOR_NOT_READY', 'message': 'Audio not initialised'})
        return

    source_id = data.get('source_id')
    filename = data.get('filename')

    if not source_id or not filename:
        emit('error', {'message': 'source_id and filename required'})
        return

    try:
        session_dir = os.path.join(source_manager.upload_dir, session_id)
        file_path = os.path.join(session_dir, secure_filename(filename))

        if not os.path.exists(file_path):
            emit('error', {'message': 'File not found'})
            return

        t0 = time.time()
        buffers = decoded_buffer_cache.get(file_path)
        if buffers is None:
            emit('status', {'code': 'DECODING', 'message': f'Decoding {filename}…', 'source_id': source_id})
            buffers = source_manager.convert_mp3_to_audio_buffers(file_path, config.SAMPLE_RATE, config.BUFFER_SIZE)
            if buffers:
                decoded_buffer_cache[file_path] = buffers
        logging.info(f"[stream] decode/cache: {(time.time()-t0)*1000:.1f}ms, {len(buffers) if buffers else 0} buffers")

        if not buffers:
            emit('error', {'message': 'Failed to decode audio file'})
            return

        if session_id not in _g_stream_locks:
            _g_stream_locks[session_id] = threading.Lock()

        with _g_stream_locks[session_id]:
            if session_id not in _g_stream_states:
                _g_stream_states[session_id] = {}
            _g_stream_states[session_id][source_id] = {'buffers': buffers, 'pos': 0}

        with _g_lock:
            _g_stream_active.add(session_id)

        logging.info(f"[stream] registered in global loop: {(time.time()-t0)*1000:.1f}ms")

        emit('status', {
            'code': 'FILE_STREAM_STARTED',
            'source_id': source_id,
            'filename': filename,
            'total_buffers': len(buffers),
            'server_ts': int(time.time() * 1000),
        })

    except Exception as e:
        logging.error(f"stream_uploaded_file error: {e}")
        emit('error', {'message': f'File streaming failed: {str(e)}'})


def _do_render_session(session_id: str, buffers_map: dict):
    """render one tick for session_id; called from _render_pool threads."""
    processor = audio_processors.get(session_id)
    if not processor or not processor.is_initialised:
        return session_id, None, None
    result = processor.process_sources(buffers_map)
    if result is None:
        return session_id, None, None
    return session_id, result[0], result[1]


def _global_render_loop():
    """gevent greenlet: drives all active sessions at real-time pace."""
    buffer_duration = config.BUFFER_SIZE / config.SAMPLE_RATE
    t_start = time.time()
    tick = 0

    while True:
        tick += 1
        try:
            with _g_lock:
                active_sessions = list(_g_stream_active)

            futures_map = {}        # future → (session_id, source_positions)
            completed_sessions = [] # sessions that finished all tracks this tick

            for session_id in active_sessions:
                lock  = _g_stream_locks.get(session_id)
                state = _g_stream_states.get(session_id)
                if lock is None or state is None:
                    continue

                buffers_map      = {}
                source_positions = {}
                track_finished   = []

                with lock:
                    paused = _g_stream_paused.get(session_id, set())
                    for src_id, src in list(state.items()):
                        if src_id in paused:
                            continue
                        pos   = src['pos']
                        total = len(src['buffers'])
                        if pos >= total:
                            track_finished.append(src_id)
                            continue
                        buffers_map[src_id]      = src['buffers'][pos]
                        src['pos']               = pos + 1
                        source_positions[src_id] = {'current': pos, 'total': total}
                    for src_id in track_finished:
                        del state[src_id]
                    session_done = len(state) == 0

                for src_id in track_finished:
                    socketio.emit('status', {
                        'code': 'TRACK_COMPLETE',
                        'source_id': src_id,
                        'server_ts': int(time.time() * 1000),
                    }, to=session_id, namespace='/')

                if session_done:
                    completed_sessions.append(session_id)
                    continue

                if not buffers_map:
                    continue  # all sources paused

                ar = _render_pool.apply_async(_do_render_session, (session_id, buffers_map))
                futures_map[ar] = (session_id, source_positions)

            # collect results — .get() suspends this greenlet (not the event loop) while py3dti runs
            for ar, (session_id, source_positions) in futures_map.items():
                try:
                    r_sid, left, right = ar.get()
                    if left is not None:
                        socketio.emit('file_audio_chunk', {
                            'audio_data': encode_audio_to_base64(left, right),
                            'source_positions': source_positions,
                            'render_ts': int(time.time() * 1000),
                        }, to=r_sid, namespace='/')
                except Exception as e:
                    logging.error(f"render error for {session_id}: {e}")

            # emit STREAM_COMPLETE and clean up sessions that finished naturally
            for session_id in completed_sessions:
                with _g_lock:
                    _g_stream_active.discard(session_id)
                _g_stream_states.pop(session_id, None)
                _g_stream_locks.pop(session_id, None)
                _g_stream_paused.pop(session_id, None)
                socketio.emit('status', {
                    'code': 'STREAM_COMPLETE',
                    'server_ts': int(time.time() * 1000),
                }, to=session_id, namespace='/')

        except Exception as e:
            logging.error(f"render loop tick {tick} error: {e}", exc_info=True)

        t_next = t_start + tick * buffer_duration
        sleep_time = t_next - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif sleep_time < -10 * buffer_duration:
            t_start = time.time() - tick * buffer_duration


@socketio.on('stop_file_stream')
def handle_stop_file_stream():
    """stop all tracks for this session."""
    _stop_file_stream(request.sid)
    emit('status', {'code': 'STREAM_STOPPED', 'server_ts': int(time.time() * 1000)})


@socketio.on('pause_stream')
def handle_pause_stream(data=None):
    session_id = request.sid
    src_id = (data or {}).get('source_id')
    if src_id:
        _g_stream_paused.setdefault(session_id, set()).add(src_id)
    emit('status', {'code': 'STREAM_PAUSED', 'source_id': src_id, 'server_ts': int(time.time() * 1000)})


@socketio.on('resume_stream')
def handle_resume_stream(data=None):
    session_id = request.sid
    src_id = (data or {}).get('source_id')
    if src_id and session_id in _g_stream_paused:
        _g_stream_paused[session_id].discard(src_id)
    emit('status', {'code': 'STREAM_RESUMED', 'source_id': src_id, 'server_ts': int(time.time() * 1000)})


@socketio.on('stop_source_stream')
def handle_stop_source_stream(data=None):
    session_id = request.sid
    src_id = (data or {}).get('source_id')
    if not src_id:
        return
    lock = _g_stream_locks.get(session_id)
    if lock:
        with lock:
            _g_stream_states.get(session_id, {}).pop(src_id, None)
    if session_id in _g_stream_paused:
        _g_stream_paused[session_id].discard(src_id)
    emit('status', {'code': 'SOURCE_STOPPED', 'source_id': src_id, 'server_ts': int(time.time() * 1000)})


@socketio.on('batch_update_positions')
def handle_batch_update_positions(data):
    """batch multiple source-position updates in a single message."""
    session_id = request.sid
    processor = audio_processors.get(session_id)
    updates = (data or {}).get('updates', [])

    results = []
    for u in updates:
        src_id  = u.get('source_id')
        pos     = u.get('position')
        if src_id and pos and processor:
            ok = processor.update_source_position(src_id, pos)
            results.append({'source_id': src_id, 'ok': ok})
        else:
            results.append({'source_id': src_id, 'ok': False, 'error': 'missing fields or no processor'})

    emit('status', {
        'code': 'BATCH_POSITIONS_UPDATED',
        'results': results,
        'server_ts': int(time.time() * 1000),
    })


@socketio.on('list_uploaded_files')
def handle_list_uploaded_files():
    """List all uploaded files for the current session"""
    session_id = request.sid
    session = session_manager.get_session(session_id)
    
    if not session:
        emit('error', {'message': 'Session not found'})
        return
    
    try:
        files = source_manager.get_session_files(session_id)
        emit('status', {
            'code': 'FILES_LISTED',
            'message': f'Found {len(files)} uploaded files',
            'files': files,
            'total_files': len(files)
        })
        
    except Exception as e:
        logging.error(f"List files error: {e}")
        emit('error', {'message': f'Failed to list files: {str(e)}'})

_render_loop_thread = threading.Thread(target=_global_render_loop, daemon=True, name='global-render-loop')
_render_loop_thread.start()

if __name__ == '__main__':
    print(f"Starting WebSocket server...")
    socketio.run(
        app,
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
    )