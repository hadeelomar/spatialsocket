import os

class Config:
    """application configuration."""
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    SECRET_KEY = os.environ.get('SECRET_KEY')

    SAMPLE_RATE = int(os.environ.get('SAMPLE_RATE', 48000))
    BUFFER_SIZE = int(os.environ.get('BUFFER_SIZE', 1024))
    MAX_SOURCES_PER_SESSION = int(os.environ.get('MAX_SOURCES_PER_SESSION', 16))

    SESSION_TIMEOUT_SECONDS = int(os.environ.get('SESSION_TIMEOUT_SECONDS', 86400))
    SESSION_CLEANUP_INTERVAL = int(os.environ.get('SESSION_CLEANUP_INTERVAL', 60))

    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    HRTF_DATASETS_DIR = os.environ.get('HRTF_DATASETS_DIR', os.path.join(_PROJECT_ROOT, 'hrtf_datasets'))
    DEFAULT_HRTF_FILE = os.environ.get('DEFAULT_HRTF_FILE', 'p0200.sofa')
    HRTF_UPLOAD_DIR = os.environ.get('HRTF_UPLOAD_DIR', os.path.join(_PROJECT_ROOT, 'hrtf_uploads'))

    CORS_ALLOWED_ORIGINS = '*'

config = Config()