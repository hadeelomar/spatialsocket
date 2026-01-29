import os

class Config:
    """Base configuration"""
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    SECRET_KEY = os.environ.get('SECRET_KEY')
    
    # Audio settings
    SAMPLE_RATE = int(os.environ.get('SAMPLE_RATE', 48000))
    BUFFER_SIZE = int(os.environ.get('BUFFER_SIZE', 1024))
    MAX_SOURCES_PER_SESSION = int(os.environ.get('MAX_SOURCES_PER_SESSION', 16))
    
    # Session settings
    SESSION_TIMEOUT_SECONDS = int(os.environ.get('SESSION_TIMEOUT_SECONDS', 300))  # 5 minutes
    SESSION_CLEANUP_INTERVAL = int(os.environ.get('SESSION_CLEANUP_INTERVAL', 60))  # 1 minute
    
    # CORS settings
    CORS_ALLOWED_ORIGINS = '*'

config = Config()