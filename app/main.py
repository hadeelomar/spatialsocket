from dotenv import load_dotenv
load_dotenv()
import time
import logging
import atexit
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from typing import Dict, Tuple, Optional
from config import config
from session_manager import SessionManager, StreamState
from audio_processor import AudioProcessor
from audio_streamer import (
    encode_audio_to_base64,
    decode_audio_from_base64,
    generate_test_tone
)
from performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
CORS(app, resources={r"/*": {"origins": config.CORS_ALLOWED_ORIGINS}})

socketio = SocketIO(
    app,
    cors_allowed_origins=config.CORS_ALLOWED_ORIGINS,
    async_mode='threading'
)

# Initialise managers with config values
session_manager = SessionManager(
    session_timeout=config.SESSION_TIMEOUT_SECONDS,
    cleanup_interval=config.SESSION_CLEANUP_INTERVAL
)
performance_monitor = PerformanceMonitor()
audio_processors: Dict[str, AudioProcessor] = {}

# Register shutdown handler
atexit.register(shutdown_server)

def shutdown_server():
    """Clean shutdown of server resources"""
    logging.info("Shutting down server...")
    
    # Cleanup all audio processors
    for session_id, processor in list(audio_processors.items()):
        try:
            processor.cleanup()
            logging.info(f"Cleaned up audio processor for session {session_id}")
        except Exception as e:
            logging.error(f"Error cleaning up processor for {session_id}: {e}")
    audio_processors.clear()
    
    # Shutdown session manager
    session_manager.shutdown()
    
    logging.info("Server shutdown complete")

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

@app.route('/sessions')
def get_sessions():
    """Get information about all active sessions"""
    return jsonify({
        'sessions': session_manager.get_all_session_info(),
        'total_sessions': session_manager.get_session_count()
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    logging.info(f"Client connected: {session_id}")
    
    performance_monitor.record_receive_timestamp(session_id)
    
    # Create session with config parameters
    session = session_manager.create_session(
        session_id, 
        config.SAMPLE_RATE, 
        config.BUFFER_SIZE
    )
    
    # Initialise audio processor
    processor = AudioProcessor(config.SAMPLE_RATE, config.BUFFER_SIZE, config.MAX_SOURCES_PER_SESSION)
    audio_processors[session_id] = processor
    
    session.set_stream_state(StreamState.IDLE)
    performance_monitor.record_send_timestamp(session_id)
    
    emit('connected', {
        'session_id': session_id,
        'sample_rate': config.SAMPLE_RATE,
        'buffer_size': config.BUFFER_SIZE,
        'message': 'Connected to SpatialSocket API'
    })
    
    logging.info(f"Session {session_id} fully initialised")

@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    logging.info(f"Client disconnected: {session_id}")
    performance_monitor.record_receive_timestamp(session_id)
    
    # Get session before cleanup
    session = session_manager.get_session(session_id)
    if session:
        session.set_stream_state(StreamState.STOPPING)
    
    # Clean up audio processor
    if session_id in audio_processors:
        try:
            processor = audio_processors[session_id]
            processor.cleanup()
            del audio_processors[session_id]
            logging.info(f"Audio processor cleaned up for session {session_id}")
        except Exception as e:
            logging.error(f"Error cleaning up processor for {session_id}: {e}")
    
    # Remove session from manager
    session_manager.remove_session(session_id)
    performance_monitor.cleanup_session(session_id)
    
    logging.info(f"Session {session_id} fully disconnected and cleaned up")

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
            # Update metrics
            performance_monitor.decrement_source_count(session_id)
        
        emit('status', {
            'code': 'SOURCE_REMOVED' if success else 'REMOVE_FAILED',
            'source_id': source_id,
            'remaining_sources': processor.get_source_count()
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
        if success:
            # Update metrics
            performance_monitor.increment_source_count(session_id)
        
        emit('status', {
            'code': 'SOURCE_CREATED' if success else 'CREATE_FAILED',
            'source_id': source_id,
            'position': position,
            'total_sources': processor.get_source_count()
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


@socketio.on('update_listener')
def handle_update_listener(data):
    """Handle listener pose updates with validation and optimisation."""
    session_id = request.sid
    processor = audio_processors.get(session_id)
    session = session_manager.get_session(session_id)
    
    performance_monitor.record_receive_timestamp(session_id)
    performance_monitor.record_processing_start(session_id)
    
    # Validate payload structure
    validation_error = _validate_listener_payload(data)
    if validation_error:
        performance_monitor.increment_errors()
        emit('error', {
            'code': 'INVALID_PAYLOAD',
            'message': validation_error
        })
        return
    
    # Extract validated data
    position = data['position']
    orientation = data['orientation']
    
    if not session:
        performance_monitor.increment_errors()
        emit('error', {
            'code': 'SESSION_NOT_FOUND',
            'message': 'Session not found'
        })
        return
    
    # Update session listener pose with optimisation tracking
    position_changed = session.update_listener_pose(position, orientation)
    
    # Update audio processor if available
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
        'pose': session.get_listener_pose()
    })


def _validate_listener_payload(data: dict) -> str:
    """
    Validate listener update payload.
    
    Args:
        data: Incoming payload data
        
    Returns:
        str: Error message if validation fails, None if valid
    """
    # Check required top-level keys
    required_keys = ['position', 'orientation']
    for key in required_keys:
        if key not in data:
            return f'Missing required key: {key}'
    
    position = data['position']
    orientation = data['orientation']
    
    # Validate position
    if not isinstance(position, dict):
        return 'Position must be a dictionary'
    
    pos_required_keys = ['x', 'y', 'z']
    for key in pos_required_keys:
        if key not in position:
            return f'Missing required position key: {key}'
        if not isinstance(position[key], (int, float)):
            return f'Position {key} must be a number'
        # Check for reasonable ranges (e.g., within +/-1000 meters)
        if abs(position[key]) > 1000:
            return f'Position {key} value {position[key]} is outside reasonable range (-1000 to 1000)'
    
    # Validate orientation structure
    if not isinstance(orientation, dict):
        return 'Orientation must be a dictionary'
    
    orient_required_keys = ['forward', 'up']
    for key in orient_required_keys:
        if key not in orientation:
            return f'Missing required orientation key: {key}'
        if not isinstance(orientation[key], dict):
            return f'Orientation {key} must be a dictionary'
        
        # Validate vector components
        vector = orientation[key]
        vector_required_keys = ['x', 'y', 'z']
        for vkey in vector_required_keys:
            if vkey not in vector:
                return f'Missing required orientation {key} key: {vkey}'
            if not isinstance(vector[vkey], (int, float)):
                return f'Orientation {key} {vkey} must be a number'
            # Check for unit vector ranges (-1 to 1)
            if abs(vector[vkey]) > 1.1:  # Small tolerance for floating point
                return f'Orientation {key} {vkey} value {vector[vkey]} is outside unit vector range (-1 to 1)'
    
    # Validate that forward and up vectors are not parallel (cross product should not be zero)
    forward = orientation['forward']
    up = orientation['up']
    
    # Simple check: vectors should not be identical or opposite
    dot_product = (forward['x'] * up['x'] + forward['y'] * up['y'] + forward['z'] * up['z'])
    if abs(dot_product) > 0.99:  # Nearly parallel
        return 'Forward and up vectors should not be parallel'
    
    return None  # Validation passed


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
    
    result = processor.process_audio(source_id, audio_buffer)
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