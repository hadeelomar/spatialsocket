import time
from typing import Dict, Optional
from threading import Lock

class Session:
    """ Represents a single client session """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_activity = time.time()
        self.data = {}

    def touch(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def is_expired(self, timeout: int) -> bool:
        """Check if session has been inactive too long"""
        return (time.time() - self.last_activity) > timeout

class SessionManager:
    """Manages all client sessions"""

    def __init__(self, session_timeout: int=300):
        self.sessions: Dict[str, Session] = {}
        self.session_timeout = session_timeout
        self.lock = Lock()
        print("SessionManager initialised")
    
    def create_session(self, session_id: str) -> Session:
        """Create a new session"""
        with self.lock:
            session = Session(session_id)
            self.sessions[session_id] = session
            print(f"Session created: {session_id}")
            return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        with self.lock:
            return self.sessions.get(session_id)
        
    def remove_session(self, session_id: str):
        """Remove a session"""
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                print(f"Session removed: {session_id}")
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        with self.lock:
            return len(self.sessions)