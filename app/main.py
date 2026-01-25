from dotenv import load_dotenv
load_dotenv()
import time
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from typing import Dict, Tuple, Optional
from config import config
from session_manager import SessionManager
from audio_processor import AudioProcessor
from audio_streamer import (
    encode_audio_to_base64,
    decode_audio_from_base64,
    generate_test_tone
)
from performance_monitor import PerformanceMonitor

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
CORS(app, resources={r"/*": {"origins": config.CORS_ALLOWED_ORIGINS}})

socketio = SocketIO(
    app,
    cors_allowed_origins=config.CORS_ALLOWED_ORIGINS,
    async_mode='threading'
)

session_manager = SessionManager()
performance_monitor = PerformanceMonitor()
audio_processors: Dict[str, AudioProcessor] = {}

@app.route('/')
def index():
    """ Root endpoint - health check """
    return jsonify({
        'service': 'SpatialSocket API',
        'version': '0.4.0',
        'status': 'running',
        'active_sessions': session_manager.get_session_count(),
        'performance_metrics': performance_monitor.to_dict()['global']
    })

@app.route('/metrics')
def get_metrics():
    """Get performance metrics endpoint"""
    return jsonify(performance_monitor.to_dict())

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    print(f"Client connected: {request.sid}")
    performance_monitor.record_receive_timestamp(session_id)
    session = session_manager.create_session(session_id)
    processor = AudioProcessor(config.SAMPLE_RATE, config.BUFFER_SIZE)
    audio_processors[session_id] = processor
    performance_monitor.record_send_timestamp(session_id)
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
    performance_monitor.record_receive_timestamp(session_id)

    if session_id in audio_processors:
        audio_processors[session_id].cleanup()
        del audio_processors[session_id]
    
    session_manager.remove_session(session_id)
    performance_monitor.cleanup_session(session_id)

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


@socketio.on('stream_audio')
def handle_stream_audio(data):
    session_id = request.sid
    processor = audio_processors.get(session_id)
    
    performance_monitor.record_receive_timestamp(session_id)
    performance_monitor.record_processing_start(session_id)

    if not processor or not processor.is_initialised:
        performance_monitor.increment_errors()
        emit('error', {
            'code': 'PROCESSOR_NOT_READY',
            'message': 'Audio processor not initalised'
        })
        return
    
    source_id = data.get('source_id')
    audio_data_encoded = data.get('audio_data')

    if not source_id or not audio_data_encoded:
        performance_monitor.increment_errors()
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
        emit('error', {
            'code': 'DECODE_ERROR',
            'message': 'Failed to decode audio data'
        })
        return
    
    result = processor.process_audio(source_id, audio_buffer)
    if result is None:
        performance_monitor.increment_errors()
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
        # Validate session exists
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
        
        # Ensure source exists, create if missing
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

            result = processor.process_audio(source_id, chunk)

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
    session_id = request.sid
    performance_monitor.record_receive_timestamp(session_id)
    session = session_manager.get_session(session_id)

    if session:
        session.touch()
        performance_monitor.record_send_timestamp(session_id)
        emit('status', {'message': 'Data retrieved', 'data': session.data})
    else:
        performance_monitor.increment_errors()
        emit('error', {'message': 'Session not found'})

if __name__ == '__main__':
    print(f"Starting WebSocket server...")
    socketio.run(
        app,
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )