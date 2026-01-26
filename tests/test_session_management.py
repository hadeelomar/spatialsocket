"""
Unit tests for session management
"""

import unittest
import time
import threading
import logging
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from session_manager import SessionManager, Session, StreamState
from config import config


class TestSession(unittest.TestCase):
    """Test cases for the Session class"""
    
    def setUp(self):
        self.session_id = "test_session_123"
        self.sample_rate = 48000
        self.buffer_size = 1024
        self.session = Session(self.session_id, self.sample_rate, self.buffer_size)
    
    def test_session_initialisation(self):
        """Test session is initialised with correct metadata"""
        self.assertEqual(self.session.session_id, self.session_id)
        self.assertEqual(self.session.stream_state, StreamState.IDLE)
        self.assertEqual(self.session.config['sample_rate'], self.sample_rate)
        self.assertEqual(self.session.config['buffer_size'], self.buffer_size)
        self.assertFalse(self.session.cleanup_complete)
        self.assertIsInstance(self.session.created_at, float)
        self.assertIsInstance(self.session.last_seen, float)
        self.assertIsInstance(self.session.last_activity, float)
    
    def test_update_last_seen(self):
        """Test last_seen timestamp updates"""
        original_last_seen = self.session.last_seen
        time.sleep(0.01)
        self.session.update_last_seen()
        self.assertGreater(self.session.last_seen, original_last_seen)
    
    def test_touch_method(self):
        """Test touch method updates both timestamps"""
        original_last_seen = self.session.last_seen
        original_last_activity = self.session.last_activity
        time.sleep(0.01)
        self.session.touch()
        self.assertGreater(self.session.last_seen, original_last_seen)
        self.assertGreater(self.session.last_activity, original_last_activity)
    
    def test_stream_state_changes(self):
        """Test stream state transitions"""
        self.session.set_stream_state(StreamState.STREAMING)
        self.assertEqual(self.session.stream_state, StreamState.STREAMING)
        self.session.set_stream_state(StreamState.STOPPING)
        self.assertEqual(self.session.stream_state, StreamState.STOPPING)
        self.session.set_stream_state(StreamState.ERROR)
        self.assertEqual(self.session.stream_state, StreamState.ERROR)
    
    def test_is_expired(self):
        """Test session expiry detection"""
        # Fresh session should not be expired
        self.assertFalse(self.session.is_expired(timeout=300))
        
        # Session with old last_seen should be expired
        self.session.last_seen = time.time() - 400  # 400 seconds ago
        self.assertTrue(self.session.is_expired(timeout=300))
        
        # Session with recent last_seen should not be expired
        self.session.last_seen = time.time() - 100  # 100 seconds ago
        self.assertFalse(self.session.is_expired(timeout=300))
    
    def test_get_session_info(self):
        """Test session info export"""
        info = self.session.get_session_info()
        
        required_keys = [
            'session_id', 'created_at', 'last_seen', 'last_activity',
            'stream_state', 'config', 'cleanup_complete', 'age_seconds', 'idle_seconds'
        ]
        
        for key in required_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['session_id'], self.session_id)
        self.assertEqual(info['stream_state'], StreamState.IDLE.value)
        self.assertIsInstance(info['age_seconds'], float)
        self.assertIsInstance(info['idle_seconds'], float)


class TestSessionManager(unittest.TestCase):
    """Test cases for the SessionManager class"""
    
    def setUp(self):
        self.session_timeout = 5
        self.cleanup_interval = 1
        self.manager = SessionManager(
            session_timeout=self.session_timeout,
            cleanup_interval=self.cleanup_interval
        )
    
    def tearDown(self):
        self.manager.shutdown()
    
    def test_manager_initialisation(self):
        """Test manager initialises with correct settings"""
        self.assertEqual(self.manager.session_timeout, self.session_timeout)
        self.assertEqual(self.manager.cleanup_interval, self.cleanup_interval)
        self.assertIsNotNone(self.manager.cleanup_thread)
        self.assertTrue(self.manager.cleanup_thread.is_alive())
    
    def test_create_session(self):
        """Test session creation"""
        session_id = "test_session"
        session = self.manager.create_session(session_id, 44100, 512)
        
        self.assertIsInstance(session, Session)
        self.assertEqual(session.session_id, session_id)
        self.assertEqual(session.config['sample_rate'], 44100)
        self.assertEqual(session.config['buffer_size'], 512)
        
        # Check session is stored in manager
        stored_session = self.manager.get_session(session_id)
        self.assertIsNotNone(stored_session)
        self.assertEqual(stored_session.session_id, session_id)
    
    def test_get_session_updates_last_seen(self):
        """Test getting a session updates last_seen"""
        session_id = "test_session"
        session = self.manager.create_session(session_id)
        original_last_seen = session.last_seen
        
        time.sleep(0.01)
        retrieved_session = self.manager.get_session(session_id)
        
        self.assertIsNotNone(retrieved_session)
        self.assertGreater(retrieved_session.last_seen, original_last_seen)
    
    def test_get_nonexistent_session(self):
        """Test getting non-existent session returns None"""
        session = self.manager.get_session("nonexistent")
        self.assertIsNone(session)
    
    def test_remove_session(self):
        """Test session removal"""
        session_id = "test_session"
        self.manager.create_session(session_id)
        
        # Verify session exists
        self.assertIsNotNone(self.manager.get_session(session_id))
        self.assertEqual(self.manager.get_session_count(), 1)
        
        # Remove session
        result = self.manager.remove_session(session_id)
        self.assertTrue(result)
        
        # Verify session is removed
        self.assertIsNone(self.manager.get_session(session_id))
        self.assertEqual(self.manager.get_session_count(), 0)
    
    def test_remove_nonexistent_session(self):
        """Test removing non-existent session returns False"""
        result = self.manager.remove_session("nonexistent")
        self.assertFalse(result)
    
    def test_cleanup_expired_sessions(self):
        """Test expired session cleanup"""
        # Create sessions
        session1 = self.manager.create_session("session1")
        session2 = self.manager.create_session("session2")
        session3 = self.manager.create_session("session3")
        
        # Make one session expired
        session1.last_seen = time.time() - 10  # 10 seconds ago
        
        # Clean up expired sessions (timeout is 5 seconds)
        expired = self.manager.cleanup_expired_sessions()
        
        # Verify expired session was removed
        self.assertIn("session1", expired)
        self.assertEqual(len(expired), 1)
        self.assertIsNone(self.manager.get_session("session1"))
        self.assertIsNotNone(self.manager.get_session("session2"))
        self.assertIsNotNone(self.manager.get_session("session3"))
    
    def test_get_all_session_info(self):
        """Test getting all session information"""
        # Create sessions with different configs
        self.manager.create_session("session1", 48000, 1024)
        self.manager.create_session("session2", 44100, 512)
        
        all_info = self.manager.get_all_session_info()
        
        self.assertEqual(len(all_info), 2)
        
        # Check info structure
        for info in all_info:
            self.assertIn('session_id', info)
            self.assertIn('config', info)
            self.assertIn('stream_state', info)
    
    def test_shutdown(self):
        """Test manager shutdown"""
        # Create a session first
        self.manager.create_session("test_session")
        
        # Shutdown the manager
        self.manager.shutdown()
        
        # Verify cleanup thread is stopped
        self.assertFalse(self.manager.cleanup_thread.is_alive())
        
        # Verify all sessions are cleared
        self.assertEqual(self.manager.get_session_count(), 0)


class TestSessionIntegration(unittest.TestCase):
    """Integration tests for session management"""
    
    def setUp(self):
        self.manager = SessionManager(session_timeout=2, cleanup_interval=1)
    
    def tearDown(self):
        self.manager.shutdown()
    
    def test_session_lifecycle(self):
        """Test complete session lifecycle"""
        session_id = "lifecycle_test"
        
        # Create session
        session = self.manager.create_session(session_id)
        self.assertEqual(session.stream_state, StreamState.IDLE)
        
        # Simulate activity
        session.set_stream_state(StreamState.STREAMING)
        self.manager.get_session(session_id)  # Updates last_seen
        
        # Simulate disconnection
        session.set_stream_state(StreamState.STOPPING)
        removed = self.manager.remove_session(session_id)
        self.assertTrue(removed)
        
        # Verify cleanup
        self.assertIsNone(self.manager.get_session(session_id))
    
    def test_automatic_expiry(self):
        """Test automatic session expiry via cleanup thread"""
        session_id = "expiry_test"
        self.manager.create_session(session_id)
        
        # Make session expired
        session = self.manager.get_session(session_id)
        session.last_seen = time.time() - 3  # 3 seconds ago (timeout is 2)
        
        # Wait for cleanup thread to run (cleanup interval is 1 second)
        time.sleep(1.5)
        
        # Session should be cleaned up
        self.assertIsNone(self.manager.get_session(session_id))
    
    @patch('time.time')
    def test_concurrent_access(self, mock_time):
        """Test thread safety with concurrent access"""
        mock_time.return_value = 1000.0
        
        # Create sessions from multiple threads
        def create_sessions(start_id, count):
            for i in range(count):
                session_id = f"session_{start_id + i}"
                self.manager.create_session(session_id)
                # Simulate some work
                time.sleep(0.001)
        
        threads = []
        thread_count = 5
        sessions_per_thread = 10
        
        for i in range(thread_count):
            thread = threading.Thread(
                target=create_sessions,
                args=(i * sessions_per_thread, sessions_per_thread)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all sessions were created
        expected_count = thread_count * sessions_per_thread
        self.assertEqual(self.manager.get_session_count(), expected_count)


class TestConfigIntegration(unittest.TestCase):
    """Test configuration integration"""
    
    def test_config_values(self):
        """Test config values are properly loaded"""
        self.assertIsInstance(config.SESSION_TIMEOUT_SECONDS, int)
        self.assertIsInstance(config.SESSION_CLEANUP_INTERVAL, int)
        self.assertGreater(config.SESSION_TIMEOUT_SECONDS, 0)
        self.assertGreater(config.SESSION_CLEANUP_INTERVAL, 0)
    
    def test_manager_uses_config(self):
        """Test manager uses config values when not explicitly set"""
        manager = SessionManager()
        self.assertEqual(manager.session_timeout, config.SESSION_TIMEOUT_SECONDS)
        self.assertEqual(manager.cleanup_interval, config.SESSION_CLEANUP_INTERVAL)
        manager.shutdown()


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    
    unittest.main(verbosity=2)
