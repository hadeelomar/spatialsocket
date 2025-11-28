from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify
from app.config import config

app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY

@app.route('/')
def index():
    """ Root endpoint - health check """
    return jsonify({
        'service': 'SpatialSocket API',
        'version': '0.2.0',
        'status': 'running',
        'sample_rate': config.SAMPLE_RATE,
        'buffer_size': config.BUFFER_SIZE
    })

@app.route('/health')
def health():
    """ Health check endpoint """
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print(f"Starting server on {config.HOST}:{config.PORT}")
    print(f"Sample rate: {config.SAMPLE_RATE} Hz")
    print(f"Buffer size: {config.BUFFER_SIZE} samples")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)