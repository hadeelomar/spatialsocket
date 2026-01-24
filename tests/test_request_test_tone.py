"""
Unit tests for SpatialSocket request_test_tone
"""
import unittest
import socketio
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

class TestRequestTestTone(unittest.TestCase):
    """Test cases for request_test_tone functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - start server and create client"""
        cls.server_started = False
        cls.sio = socketio.Client()
        cls.test_results = {}
        cls.audio_chunks_received = []
        cls.status_messages = []
        cls.error_messages = []
        
        # Set up event handlers
        @cls.sio.event
        def connect():
            cls.test_results['connected'] = True
            
        @cls.sio.event
        def connected(data):
            cls.test_results['session_data'] = data
            
        @cls.sio.event
        def status(data):
            cls.status_messages.append(data)
            
        @cls.sio.event
        def processed_audio(data):
            cls.audio_chunks_received.append(data)
            
        @cls.sio.event
        def error(data):
            cls.error_messages.append(data)
            
        @cls.sio.event
        def disconnect():
            cls.test_results['disconnected'] = True
    
    def setUp(self):
        """Set up each test - clear previous results"""
        self.audio_chunks_received.clear()
        self.status_messages.clear()
        self.error_messages.clear()
        
    def test_01_server_connection(self):
        """Test that we can connect to the server"""
        try:
            self.sio.connect('http://localhost:5000')
            time.sleep(0.5)
            self.assertTrue(self.test_results.get('connected', False), "Failed to connect to server")
        except Exception as e:
            self.skipTest(f"Server not running: {e}")
    
    def test_02_audio_initialisation(self):
        """Test audio initialisation"""
        if not self.test_results.get('connected', False):
            self.skipTest("Not connected to server")
            
        time.sleep(1)
        
        print("Initialising audio...")
        self.sio.emit('init_audio')
        time.sleep(1)
        
        # Should receive successful initialisation
        success = any(msg.get('code') == 'AUDIO_INITIALISED' for msg in self.status_messages)
        self.assertTrue(success, "Audio initialisation failed")
    
    def test_03_basic_test_tone(self):
        """Test basic test tone generation"""
        if not self.test_results.get('connected', False):
            self.skipTest("Not connected to server")
            
        initial_chunk_count = len(self.audio_chunks_received)
        
        self.sio.emit('request_test_tone', {
            'source_id': 'test_basic',
            'frequency': 440,
            'duration': 0.5
        })
        
        time.sleep(2)
        
        # Should receive audio chunks
        self.assertGreater(len(self.audio_chunks_received), initial_chunk_count, 
                          "No audio chunks received")
        
        # Should receive completion status
        completion = any(msg.get('code') == 'TEST_TONE_COMPLETE' for msg in self.status_messages)
        self.assertTrue(completion, "Test tone completion not received")
    
    def test_04_source_reuse(self):
        """Test reusing the same source multiple times"""
        if not self.test_results.get('connected', False):
            self.skipTest("Not connected to server")
            
        # First request
        self.sio.emit('request_test_tone', {
            'source_id': 'test_reuse',
            'frequency': 440,
            'duration': 0.3
        })
        time.sleep(1)
        
        first_chunks = len(self.audio_chunks_received)
        self.status_messages.clear()
        
        # Second request with same source
        self.sio.emit('request_test_tone', {
            'source_id': 'test_reuse',
            'frequency': 880,
            'duration': 0.3
        })
        time.sleep(1)
        
        # Should receive more chunks and completion
        self.assertGreater(len(self.audio_chunks_received), first_chunks,
                          "No chunks received for reused source")
        
        completion = any(msg.get('code') == 'TEST_TONE_COMPLETE' for msg in self.status_messages)
        self.assertTrue(completion, "Test tone completion not received for reused source")
        
        # Should not have source creation errors
        source_errors = [msg for msg in self.error_messages 
                        if msg.get('code') == 'SOURCE_CREATION_FAILED']
        self.assertEqual(len(source_errors), 0, "Source creation error occurred")
    
    def test_05_default_source_id(self):
        """Test using default source_id"""
        if not self.test_results.get('connected', False):
            self.skipTest("Not connected to server")
            
        initial_chunk_count = len(self.audio_chunks_received)
        
        self.sio.emit('request_test_tone', {
            'frequency': 220,
            'duration': 0.3
        })
        
        time.sleep(1.5)
        
        # Should receive audio chunks
        self.assertGreater(len(self.audio_chunks_received), initial_chunk_count,
                          "No audio chunks received for default source")
        
        # Should receive completion status
        completion = any(msg.get('code') == 'TEST_TONE_COMPLETE' for msg in self.status_messages)
        self.assertTrue(completion, "Test tone completion not received")
    
    def test_06_error_handling(self):
        """Test error handling with invalid inputs"""
        if not self.test_results.get('connected', False):
            self.skipTest("Not connected to server")
            
        # Test with invalid frequency
        self.sio.emit('request_test_tone', {
            'source_id': 'test_error',
            'frequency': 'invalid',
            'duration': 0.1
        })
        
        time.sleep(1)
        
        # Should receive a structured error
        self.assertGreater(len(self.error_messages), 0, "No error received for invalid input")
        
        # Error should have proper structure
        error = self.error_messages[-1]
        self.assertIn('code', error, "Error missing 'code' field")
        self.assertIn('message', error, "Error missing 'message' field")
    
    def test_07_rapid_requests(self):
        """Test rapid successive requests"""
        if not self.test_results.get('connected', False):
            self.skipTest("Not connected to server")
            
        initial_chunk_count = len(self.audio_chunks_received)
        
        # Send multiple rapid requests
        for i in range(3):
            self.sio.emit('request_test_tone', {
                'source_id': f'rapid_test_{i}',
                'frequency': 440 + (i * 100),
                'duration': 0.2
            })
            time.sleep(0.3)  # Short delay
        
        time.sleep(2)
        
        # Should receive chunks for all requests
        self.assertGreater(len(self.audio_chunks_received), initial_chunk_count,
                          "No chunks received for rapid requests")
        
        # Should have multiple completion messages
        completions = [msg for msg in self.status_messages 
                      if msg.get('code') == 'TEST_TONE_COMPLETE']
        self.assertGreaterEqual(len(completions), 2, "Not enough completion messages")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up - disconnect client"""
        if cls.sio.connected:
            cls.sio.disconnect()

class TestServerStability(unittest.TestCase):
    """Test server stability and robustness"""
    
    def test_server_health_check(self):
        """Test server health endpoint"""
        import requests
        try:
            response = requests.get('http://localhost:5000/health', timeout=2)
            self.assertEqual(response.status_code, 200, "Health check failed")
            data = response.json()
            self.assertEqual(data.get('status'), 'healthy', "Health status not healthy")
        except requests.exceptions.RequestException:
            self.skipTest("Server not accessible")

if __name__ == '__main__':
    suite = unittest.TestSuite()
    
    suite.addTest(unittest.makeSuite(TestRequestTestTone))
    suite.addTest(unittest.makeSuite(TestServerStability))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
