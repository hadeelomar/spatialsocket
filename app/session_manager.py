import time
import uuid
import logging
import threading
from typing import Dict, Optional, Any, List
from threading import Lock, RLock
from enum import Enum

class StreamState(Enum):
    """session streaming states"""
    IDLE = "idle"
    STREAMING = "streaming"
    STOPPING = "stopping"
    ERROR = "error"

class Session:
    """represents a single client session"""

    def __init__(self, session_id: str, sample_rate: int = 48000, buffer_size: int = 1024):
        self.session_id = session_id
        self.recovery_token = str(uuid.uuid4())
        self.created_at = time.time()
        self.last_seen = time.time()
        self.last_activity = time.time()
        self.stream_state = StreamState.IDLE
        self.data = {}
        
        self.config = {
            'sample_rate': sample_rate,
            'buffer_size': buffer_size
        }
        
        self.listener_pose = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {
                'forward': {'x': 0.0, 'y': 0.0, 'z': -1.0},
                'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
            }
        }

        self.last_listener_update_time = 0.0
        self.listener_position_changed = True  # forces update on first set_listener_pose call

        self._pose_lock = RLock()

        self.cleanup_complete = False
        
        logging.info(f"Session created: {session_id} at {self.created_at}")

    def touch(self):
        """update last activity and last seen timestamps."""
        current_time = time.time()
        self.last_activity = current_time
        self.last_seen = current_time
    
    def update_last_seen(self):
        """update only last_seen timestamp."""
        self.last_seen = time.time()
    
    def set_stream_state(self, state: StreamState):
        """update stream state with logging."""
        old_state = self.stream_state
        self.stream_state = state
        logging.info(f"Session {self.session_id} stream state: {old_state.value} -> {state.value}")
    
    def update_listener_pose(self, position: Dict[str, float], orientation: Dict[str, Dict[str, float]]) -> bool:
        """update listener pose; returns true if position changed."""
        with self._pose_lock:
            old_pos = self.listener_pose['position']
            position_changed = (
                abs(old_pos['x'] - position['x']) > 0.001 or
                abs(old_pos['y'] - position['y']) > 0.001 or
                abs(old_pos['z'] - position['z']) > 0.001
            )

            self.listener_pose['position'] = position.copy()
            self.listener_pose['orientation'] = {
                'forward': orientation['forward'].copy(),
                'up':      orientation['up'].copy()
            }

            self.listener_position_changed = position_changed
            self.last_listener_update_time = time.time()

            return position_changed
    
    def get_listener_pose(self) -> Dict[str, Any]:
        """get current listener pose (thread-safe snapshot)."""
        with self._pose_lock:
            return {
                'position': self.listener_pose['position'].copy(),
                'orientation': {
                    'forward': self.listener_pose['orientation']['forward'].copy(),
                    'up':      self.listener_pose['orientation']['up'].copy(),
                }
            }
    
    def is_expired(self, timeout: int) -> bool:
        """check if session has been inactive too long."""
        return (time.time() - self.last_seen) > timeout
    
    def get_session_info(self) -> Dict[str, Any]:
        """get session information for logging/debugging."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'last_seen': self.last_seen,
            'last_activity': self.last_activity,
            'stream_state': self.stream_state.value,
            'config': self.config,
            'listener_pose': self.listener_pose,
            'listener_position_changed': self.listener_position_changed,
            'last_listener_update_time': self.last_listener_update_time,
            'cleanup_complete': self.cleanup_complete,
            'age_seconds': time.time() - self.created_at,
            'idle_seconds': time.time() - self.last_seen
        }

class SessionManager:
    """manages all client sessions."""

    def __init__(self, session_timeout: int = 300, cleanup_interval: int = 60):
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = session_timeout
        self.cleanup_interval = cleanup_interval
        self.lock = Lock()
        self.cleanup_thread = None
        self.cleanup_stop_event = threading.Event()
        
        self._start_cleanup_thread()
        
        logging.info(f"SessionManager initialised with timeout={session_timeout}s, cleanup_interval={cleanup_interval}s")
    
    def _start_cleanup_thread(self):
        """start the background cleanup thread."""
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logging.info("Session cleanup thread started")
    
    def _cleanup_loop(self):
        """background loop for cleaning up expired sessions."""
        while not self.cleanup_stop_event.wait(self.cleanup_interval):
            try:
                expired_sessions = self.cleanup_expired_sessions()
                if expired_sessions:
                    logging.info(f"Cleaned up {len(expired_sessions)} expired sessions: {expired_sessions}")
            except Exception as e:
                logging.error(f"Error in session cleanup loop: {e}")
    
    def create_session(self, session_id: str, sample_rate: int = 48000, buffer_size: int = 1024) -> Session:
        """create a new session."""
        with self.lock:
            session = Session(session_id, sample_rate, buffer_size)
            self.sessions[session_id] = session
            logging.info(f"Session created: {session_id}")
            return session

    def find_by_recovery_token(self, token: str) -> Optional[Session]:
        """return the session whose recovery_token matches, or none."""
        with self.lock:
            for session in self.sessions.values():
                if session.recovery_token == token:
                    return session
            return None

    def migrate_session(self, old_id: str, new_id: str) -> Optional[Session]:
        """re-key a session from old_id to new_id after a socket reconnect."""
        with self.lock:
            session = self.sessions.pop(old_id, None)
            if session is None:
                return None
            session.session_id = new_id
            session.touch()
            self.sessions[new_id] = session
            logging.info(f"Session migrated: {old_id} → {new_id}")
            return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """get session by id and update last_seen."""
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.update_last_seen()
            return session
        
    def remove_session(self, session_id: str) -> bool:
        """remove a session and return success status."""
        with self.lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                session_info = session.get_session_info()
                del self.sessions[session_id]
                logging.info(f"Session removed: {session_id}, info: {session_info}")
                return True
            return False
    
    def cleanup_expired_sessions(self) -> List[str]:
        """remove all expired sessions and return list of removed session ids."""
        expired_sessions = []
        current_time = time.time()
        
        with self.lock:
            sessions_to_remove = []
            for session_id, session in self.sessions.items():
                if session.is_expired(self.session_timeout):
                    sessions_to_remove.append(session_id)
                    session_info = session.get_session_info()
                    logging.info(f"Session expired: {session_id}, idle for {current_time - session.last_seen:.1f}s")
            
            for session_id in sessions_to_remove:
                del self.sessions[session_id]
                expired_sessions.append(session_id)
        
        return expired_sessions
    
    def get_session_count(self) -> int:
        """get number of active sessions."""
        with self.lock:
            return len(self.sessions)
    
    def get_all_session_info(self) -> List[Dict[str, Any]]:
        """get information about all sessions."""
        with self.lock:
            return [session.get_session_info() for session in self.sessions.values()]
    
    def shutdown(self):
        """shutdown the session manager and cleanup thread."""
        logging.info("Shutting down SessionManager...")
        if self.cleanup_thread:
            self.cleanup_stop_event.set()
            self.cleanup_thread.join(timeout=5)
            if self.cleanup_thread.is_alive():
                logging.warning("Cleanup thread did not shutdown gracefully")
            else:
                logging.info("Cleanup thread shutdown successfully")
        with self.lock:
            session_count = len(self.sessions)
            self.sessions.clear()
            logging.info(f"Cleared {session_count} sessions during shutdown")