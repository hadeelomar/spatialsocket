from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    """ Root endpoint - health check """
    return jsonify({
        'service': 'SpatialSocket API',
        'version': '0.1.0',
        'status': 'running'
    }
    )

@app.route('/health')
def health():
    """ Health check endpoint """
    return jsonify({'status': 'healthy'})