from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from typing import Dict, Tuple, Optional
from config import config
from session_manager import SessionManager
from audio_processor import AudioProcessor

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
CORS(app, resources={r"/*": {"origins": config.CORS_ALLOWED_ORIGINS}})

socketio = SocketIO(
    app,
    cors_allowed_origins=config.CORS_ALLOWED_ORIGINS,
    async_mode='threading'
)

session_manager = SessionManager()
audio_processors: Dict[str, AudioProcessor] = {}

@app.route('/')
def index():
    """ Root endpoint - health check """
    return jsonify({
        'service': 'SpatialSocket API',
        'version': '0.4.0',
        'status': 'running',
        'active_sessions': session_manager.get_session_count()
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    print(f"Client connected: {request.sid}")
    session = session_manager.create_session(session_id)
    processor = AudioProcessor(config.SAMPLE_RATE, config.BUFFER_SIZE)
    audio_processors[session_id] = processor
    emit('connected', {
        'session_id': session_id,
        'sample_rate': config.SAMPLE_RATE,
        'buffer_size': config.BUFFER_SIZE,
        'message': 'Connected to SpatialSocket API'
    }
    )

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    print(f"Client disconnected: {request.sid}")

    if session_id in audio_processors:
        audio_processors[session_id].cleanup()
        del audio_processors[session_id]
    
    session_manager.remove_session(session_id)

@socketio.on('init_audio')
def handle_init_audio():
    session_id = request.sid
    processor = audio_processors.get(session_id)


    if processor:
        success = processor.initialise()
        emit('status', {
            'code': 'AUDIO_INITIALISED' if success else 'INIT_FAILED',
            'message': 'Audio initialised' if success else 'Failed to initialise'
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
        success = processor.create_source(source_id, position)
        emit('status', {
            'code': 'SOURCE_CREATED' if success else 'CREATE_FAILED',
            'source_id': source_id,
            'position': position
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
        success = processor.update_source_position(source_id, position)
        emit('status', {
            'code': 'POSITION_UPDATED' if success else 'UPDATE_FAILED',
            'source_id': source_id})

    
@socketio.on('set_data')
def handle_set_data(data):
    session_id = request.sid
    session = session_manager.get_session(session_id)

    if session:
        session.data.update(data)
        session.touch()
        emit('status', {
            'message': 'Data stored',
            'data': session.data
        })
    else:
        emit('error', {
            'message': 'Session not found'
        })

@socketio.on('get_data')
def handle_get_data():
    session_id = request.sid
    session = session_manager.get_session(session_id)

    if session:
        session.touch()
        emit('status', {'message': 'Data retrieved', 'data': session.data})
    else:
        emit('error', {'message': 'Session not found'})

if __name__ == '__main__':
    print(f"Starting WebSocket server...")
    socketio.run(
        app,
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )