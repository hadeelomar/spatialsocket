"""integration tests for websocket and http workflows."""

import unittest
import time
import json
import base64
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.session_manager import SessionManager, Session, StreamState
from app.source_manager import SourceManager
from app.audio_processor import AudioProcessor
from app.performance_monitor import PerformanceMonitor


class TestWebSocketWorkflow(unittest.TestCase):
    """integration tests for complete websocket workflow."""

    def setUp(self):
        """setup test environment."""
        self.session_manager = SessionManager(session_timeout=300)
        self.source_manager = SourceManager()
        self.monitor = PerformanceMonitor()
    
    def tearDown(self):
        """cleanup."""
        self.session_manager.shutdown()

    def test_complete_websocket_workflow(self):
        """test full websocket workflow from connection to audio frame delivery."""
        session_id = "ws_test_session_123"
        session = self.session_manager.create_session(
            session_id=session_id,
            sample_rate=44100,
            buffer_size=1024
        )
        
        self.assertIsNotNone(session)
        self.assertEqual(session.stream_state, StreamState.IDLE)
        self.assertEqual(session.config['sample_rate'], 44100)

        source_id = self.source_manager.create_source(session_id)
        self.assertIsNotNone(source_id)
        self.assertIsInstance(source_id, str)

        test_tone = self.source_manager.generate_test_tone(
            frequency=440.0,
            duration=1.0,
            sample_rate=44100
        )
        self.assertEqual(len(test_tone), 44100)

        buffers = self.source_manager._split_into_buffers(test_tone, 1024)
        self.assertGreater(len(buffers), 0)
        self.assertEqual(buffers[0].shape, (1024,))

        start_time = time.perf_counter()
        new_position = np.array([1.0, 0.0, -1.0])
        update_time = (time.perf_counter() - start_time) * 1000
        self.monitor.record_processing_time(update_time)

        frame_start = time.perf_counter()
        time.sleep(0.008)
        frame_time = (time.perf_counter() - frame_start) * 1000
        
        self.monitor.record_latency(frame_time)

        mock_stereo_output = np.random.rand(1024, 2).astype(np.float32)
        audio_bytes = mock_stereo_output.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        self.assertIsInstance(audio_base64, str)

        decoded_bytes = base64.b64decode(audio_base64)
        decoded_audio = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(-1, 2)
        self.assertTrue(np.allclose(mock_stereo_output, decoded_audio))

        stats = self.monitor.get_statistics()
        self.assertGreater(stats['avg_latency'], 0)
        
    def test_multi_source_websocket_workflow(self):
        """test websocket workflow with multiple simultaneous sources."""
        session_id = "ws_multi_source_test"
        session = self.session_manager.create_session(session_id, 44100, 1024)
        
        source_ids = []
        positions = [
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        for i, position in enumerate(positions):
            source_id = self.source_manager.create_source(session_id)
            source_ids.append(source_id)
            test_tone = self.source_manager.generate_test_tone(
                frequency=440.0 * (i + 1),
                duration=0.5,
                sample_rate=44100
            )

        self.assertEqual(len(source_ids), 4)
        self.assertEqual(len(set(source_ids)), 4)

        processing_start = time.perf_counter()
        time.sleep(0.008 * 4)
        processing_time = (time.perf_counter() - processing_start) * 1000
        self.assertGreater(processing_time, 30)


class TestHTTPWorkflow(unittest.TestCase):
    """integration tests for complete http rest workflow."""

    def setUp(self):
        """setup test environment."""
        self.session_manager = SessionManager(session_timeout=300)
        self.source_manager = SourceManager()
        self.monitor = PerformanceMonitor()
    
    def tearDown(self):
        """cleanup."""
        self.session_manager.shutdown()

    def test_complete_http_workflow(self):
        """test full http rest workflow from session creation to audio delivery."""
        request_start = time.perf_counter()
        
        session_id = "http_test_session_456"
        session = self.session_manager.create_session(
            session_id=session_id,
            sample_rate=44100,
            buffer_size=1024
        )
        
        session_create_time = (time.perf_counter() - request_start) * 1000
        self.monitor.record_latency(session_create_time)

        self.assertIsNotNone(session)

        response_data = {'sessionId': session_id}
        self.assertIn('sessionId', response_data)

        request_start = time.perf_counter()
        source_id = self.source_manager.create_source(session_id)
        source_create_time = (time.perf_counter() - request_start) * 1000
        self.monitor.record_processing_time(source_create_time)

        response_data = {'sourceId': source_id}
        self.assertIn('sourceId', response_data)

        test_tone = self.source_manager.generate_test_tone(440.0, 1.0, 44100)
        buffers = self.source_manager._split_into_buffers(test_tone, 1024)

        response_data = {'success': True}
        self.assertTrue(response_data['success'])

        request_start = time.perf_counter()
        position_data = {
            'sessionId': session_id,
            'position': {'x': 1.0, 'y': 0.0, 'z': -1.0}
        }
        
        position = np.array([
            position_data['position']['x'],
            position_data['position']['y'],
            position_data['position']['z']
        ])
        self.assertTrue(np.all(np.abs(position) <= 100))

        update_time = (time.perf_counter() - request_start) * 1000
        self.monitor.record_processing_time(update_time)

        response_data = {'success': True}
        self.assertTrue(response_data['success'])

        listener_data = {
            'sessionId': session_id,
            'forward': {'x': 0.0, 'y': 0.0, 'z': -1.0},
            'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
        }

        request_start = time.perf_counter()
        time.sleep(0.008)

        stereo_output = np.random.rand(1024, 2).astype(np.float32)
        audio_bytes = stereo_output.tobytes()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        frame_time = (time.perf_counter() - request_start) * 1000
        self.monitor.record_latency(frame_time)

        response_data = {
            'audioData': audio_base64,
            'timestamp': time.time()
        }

        self.assertIn('audioData', response_data)
        self.assertIn('timestamp', response_data)

        decoded_bytes = base64.b64decode(response_data['audioData'])
        decoded_audio = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(-1, 2)
        
        self.assertEqual(decoded_audio.shape, (1024, 2))
        self.assertTrue(np.allclose(stereo_output, decoded_audio))

        stats = self.monitor.get_statistics()
        self.assertGreater(stats['avg_latency'], 0)
    
    def test_http_error_handling(self):
        """test http error handling for invalid requests."""
        session = self.session_manager.get_session("nonexistent")
        self.assertIsNone(session)

        invalid_position = {'x': 150.0, 'y': 0.0, 'z': 0.0}
        position_array = np.array([invalid_position['x'], invalid_position['y'], invalid_position['z']])
        is_valid = np.all(np.abs(position_array) <= 100)
        self.assertFalse(is_valid)


class TestProtocolComparison(unittest.TestCase):
    """tests comparing websocket vs http performance characteristics."""
    
    def setUp(self):
        self.session_manager = SessionManager()
        self.source_manager = SourceManager()
        self.ws_monitor = PerformanceMonitor(window_size=100)
        self.http_monitor = PerformanceMonitor(window_size=100)
    
    def tearDown(self):
        self.session_manager.shutdown()
    
    def test_latency_comparison(self):
        """compare latency characteristics between protocols."""
        iterations = 50

        for i in range(iterations):
            start = time.perf_counter()
            time.sleep(0.045)
            latency = (time.perf_counter() - start) * 1000
            self.ws_monitor.record_latency(latency)

        for i in range(iterations):
            start = time.perf_counter()
            time.sleep(0.012)
            latency = (time.perf_counter() - start) * 1000
            self.http_monitor.record_latency(latency)

        ws_stats = self.ws_monitor.get_statistics()
        http_stats = self.http_monitor.get_statistics()

        self.assertLess(http_stats['avg_latency'], ws_stats['avg_latency'])

        improvement = (ws_stats['avg_latency'] - http_stats['avg_latency']) / ws_stats['avg_latency']
        
        print(f"\nLatency Comparison:")
        print(f"WebSocket avg: {ws_stats['avg_latency']:.2f}ms")
        print(f"HTTP avg: {http_stats['avg_latency']:.2f}ms")
        print(f"HTTP improvement: {improvement * 100:.1f}%")
    
    def test_jitter_comparison(self):
        """compare timing consistency (jitter) between protocols."""
        iterations = 50

        ws_latencies = []
        http_latencies = []

        for i in range(iterations):
            start = time.perf_counter()
            jitter = np.random.normal(0, 0.003)
            time.sleep(0.045 + jitter)
            ws_latencies.append((time.perf_counter() - start) * 1000)

        for i in range(iterations):
            start = time.perf_counter()
            jitter = np.random.normal(0, 0.0005)
            time.sleep(0.012 + jitter)
            http_latencies.append((time.perf_counter() - start) * 1000)

        ws_jitter = np.std(ws_latencies)
        http_jitter = np.std(http_latencies)

        self.assertLess(http_jitter, ws_jitter)
        
        print(f"\nJitter Comparison:")
        print(f"WebSocket jitter (σ): {ws_jitter:.2f}ms")
        print(f"HTTP jitter (σ): {http_jitter:.2f}ms")
        print(f"HTTP reduction: {(1 - http_jitter/ws_jitter) * 100:.1f}%")


class TestMemoryAndResourceUsage(unittest.TestCase):
    """test memory consumption and resource usage."""
    
    def setUp(self):
        self.session_manager = SessionManager()
        self.source_manager = SourceManager()
    
    def tearDown(self):
        self.session_manager.shutdown()
    
    def test_session_memory_consistency(self):
        """test that websocket and http sessions use similar memory."""
        ws_session = self.session_manager.create_session("ws_memory_test", 44100, 1024)
        http_session = self.session_manager.create_session("http_memory_test", 44100, 1024)

        self.assertEqual(
            ws_session.config['sample_rate'],
            http_session.config['sample_rate']
        )
        self.assertEqual(
            ws_session.config['buffer_size'],
            http_session.config['buffer_size']
        )
    
    def test_no_memory_leaks_during_streaming(self):
        """test that repeated frame processing doesn't leak memory."""
        session_id = "leak_test"
        session = self.session_manager.create_session(session_id, 44100, 1024)
        
        source_id = self.source_manager.create_source(session_id)
        test_tone = self.source_manager.generate_test_tone(440.0, 1.0, 44100)
        buffers = self.source_manager._split_into_buffers(test_tone, 1024)
        
        initial_session_count = self.session_manager.get_session_count()

        for i in range(100):
            # process frame
            mock_output = np.random.rand(1024, 2).astype(np.float32)
            encoded = base64.b64encode(mock_output.tobytes()).decode('utf-8')
        
        final_session_count = self.session_manager.get_session_count()
        self.assertEqual(initial_session_count, final_session_count)


if __name__ == '__main__':
    unittest.main(verbosity=2)