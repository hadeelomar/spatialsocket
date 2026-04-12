"""unit tests for all spatialsocket core modules."""

import unittest
import time
import threading
import numpy as np
import os
import tempfile
from unittest.mock import patch, Mock
import json
import base64

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from app.session_manager import SessionManager, Session, StreamState
from app.source_manager import SourceManager
from app.audio_processor import AudioProcessor
from app.audio_streamer import AudioStreamer
from app.performance_monitor import PerformanceMonitor
from app.config import config


class TestSession(unittest.TestCase):
    """test cases for session class."""

    def setUp(self):
        self.session_id = "test_session_123"
        self.sample_rate = 44100
        self.buffer_size = 1024
        self.session = Session(self.session_id, self.sample_rate, self.buffer_size)
    
    def test_session_initialisation(self):
        """test session initialises with correct values."""
        self.assertEqual(self.session.session_id, self.session_id)
        self.assertEqual(self.session.stream_state, StreamState.IDLE)
        self.assertEqual(self.session.config['sample_rate'], self.sample_rate)
        self.assertEqual(self.session.config['buffer_size'], self.buffer_size)
        self.assertIsInstance(self.session.created_at, float)
        self.assertIsInstance(self.session.last_seen, float)
    
    def test_listener_pose_defaults(self):
        """test listener pose initializes to default values."""
        self.assertTrue(np.allclose(self.session.listener_position, [0.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(self.session.listener_forward, [0.0, 0.0, -1.0]))
        self.assertTrue(np.allclose(self.session.listener_up, [0.0, 1.0, 0.0]))
    
    def test_update_listener_pose(self):
        """test updating listener position and orientation."""
        new_position = np.array([1.0, 2.0, 3.0])
        new_forward = np.array([0.0, 0.0, -1.0])
        new_up = np.array([0.0, 1.0, 0.0])
        
        self.session.listener_position = new_position
        self.session.listener_forward = new_forward
        self.session.listener_up = new_up
        
        self.assertTrue(np.allclose(self.session.listener_position, new_position))
        self.assertTrue(np.allclose(self.session.listener_forward, new_forward))
        self.assertTrue(np.allclose(self.session.listener_up, new_up))
    
    def test_stream_state_transitions(self):
        """test stream state changes."""
        states = [
            StreamState.IDLE,
            StreamState.STREAMING,
            StreamState.STOPPING,
            StreamState.ERROR
        ]
        
        for state in states:
            self.session.set_stream_state(state)
            self.assertEqual(self.session.stream_state, state)
    
    def test_is_expired(self):
        """test session expiry detection."""
        self.assertFalse(self.session.is_expired(timeout=300))
        self.session.last_seen = time.time() - 400
        self.assertTrue(self.session.is_expired(timeout=300))
    
    def test_get_session_info(self):
        """test session info export."""
        info = self.session.get_session_info()
        
        required_keys = [
            'session_id', 'created_at', 'last_seen', 'stream_state',
            'config', 'age_seconds', 'idle_seconds'
        ]
        
        for key in required_keys:
            self.assertIn(key, info)


class TestSessionManager(unittest.TestCase):
    """test cases for sessionmanager class."""
    
    def setUp(self):
        self.manager = SessionManager(session_timeout=5, cleanup_interval=1)
    
    def tearDown(self):
        self.manager.shutdown()
    
    def test_create_session(self):
        """test session creation."""
        session_id = "test_session"
        session = self.manager.create_session(session_id, 44100, 1024)
        
        self.assertIsInstance(session, Session)
        self.assertEqual(session.session_id, session_id)
        self.assertIsNotNone(self.manager.get_session(session_id))
    
    def test_get_nonexistent_session(self):
        """test getting non-existent session returns none."""
        session = self.manager.get_session("nonexistent")
        self.assertIsNone(session)
    
    def test_remove_session(self):
        """test session removal."""
        session_id = "test_session"
        self.manager.create_session(session_id)
        
        self.assertIsNotNone(self.manager.get_session(session_id))
        result = self.manager.remove_session(session_id)
        
        self.assertTrue(result)
        self.assertIsNone(self.manager.get_session(session_id))
    
    def test_cleanup_expired_sessions(self):
        """test automatic cleanup of expired sessions."""
        session1 = self.manager.create_session("session1")
        session2 = self.manager.create_session("session2")
        session1.last_seen = time.time() - 10
        expired = self.manager.cleanup_expired_sessions()
        
        self.assertIn("session1", expired)
        self.assertIsNone(self.manager.get_session("session1"))
        self.assertIsNotNone(self.manager.get_session("session2"))
    
    def test_thread_safety(self):
        """test concurrent session creation is thread-safe."""
        def create_sessions(start_id, count):
            for i in range(count):
                session_id = f"session_{start_id}_{i}"
                self.manager.create_session(session_id)
                time.sleep(0.001)
        
        threads = [
            threading.Thread(target=create_sessions, args=(i, 10))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.manager.get_session_count(), 50)


class TestSourceManager(unittest.TestCase):
    """test cases for sourcemanager class."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.source_manager = SourceManager(upload_dir=self.temp_dir)
    
    def tearDown(self):
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_create_source(self):
        """test source creation."""
        session_id = "test_session"
        source_id = self.source_manager.create_source(session_id)
        
        self.assertIsInstance(source_id, str)
        self.assertTrue(len(source_id) > 0)
    
    def test_generate_test_tone(self):
        """test test tone generation."""
        frequency = 440.0
        duration = 1.0
        sample_rate = 44100
        
        tone = self.source_manager.generate_test_tone(
            frequency, duration, sample_rate
        )
        
        self.assertIsInstance(tone, np.ndarray)
        self.assertEqual(len(tone), int(sample_rate * duration))
        self.assertEqual(tone.dtype, np.float32)
        self.assertLessEqual(np.max(np.abs(tone)), 1.0)
    
    def test_stereo_to_mono_conversion(self):
        """test stereo to mono conversion."""
        stereo = np.random.rand(1000, 2).astype(np.float32)
        
        mono = self.source_manager._stereo_to_mono(stereo)
        
        self.assertEqual(mono.shape, (1000,))
        self.assertEqual(mono.dtype, np.float32)
        expected = stereo.mean(axis=1)
        self.assertTrue(np.allclose(mono, expected))
    
    def test_buffer_splitting(self):
        """test audio buffer splitting."""
        audio_data = np.random.rand(5000).astype(np.float32)
        buffer_size = 1024
        
        buffers = self.source_manager._split_into_buffers(audio_data, buffer_size)
        self.assertEqual(len(buffers), 5)
        for buf in buffers:
            self.assertEqual(len(buf), buffer_size)
        self.assertTrue(np.all(buffers[-1][904:] == 0.0))
    
    @patch('miniaudio.decode_file')
    def test_audio_decoding(self, mock_decode):
        """test audio file decoding."""
        mock_audio = Mock()
        mock_audio.samples = np.random.rand(44100).astype(np.float32)
        mock_audio.sample_rate = 44100
        mock_audio.num_channels = 1
        mock_decode.return_value = mock_audio
        
        session_id = "test_session"
        source_id = "test_source"
        filepath = os.path.join(self.temp_dir, f"{source_id}.mp3")
        with open(filepath, 'wb') as f:
            f.write(b'fake audio data')
        
        buffers = self.source_manager.decode_audio(session_id, source_id, filepath)
        
        self.assertIsInstance(buffers, list)
        self.assertGreater(len(buffers), 0)
    
    def test_sample_rate_conversion(self):
        """test sample rate conversion to 44100 hz."""
        original_rate = 48000
        target_rate = 44100
        duration = 1.0
        
        audio_48k = np.random.rand(int(original_rate * duration)).astype(np.float32)
        converted = self.source_manager._resample_audio(
            audio_48k, original_rate, target_rate
        )
        expected_length = int(target_rate * duration)
        self.assertAlmostEqual(len(converted), expected_length, delta=expected_length * 0.01)


class TestAudioProcessor(unittest.TestCase):
    """test cases for audioprocessor class."""

    @patch('py3dti.Core')
    def setUp(self, mock_core_class):
        """setup with mocked py3dti."""
        self.mock_core = Mock()
        mock_core_class.return_value = self.mock_core
        self.mock_core.LoadHRTF.return_value = True
        
        self.processor = AudioProcessor(hrtf_path="fake_hrtf.sofa")
    
    def test_initialization(self):
        """test processor initializes correctly."""
        self.mock_core.SetAudioState.assert_called_once()
        self.mock_core.LoadHRTF.assert_called_once()
    
    def test_hrtf_loading_failure(self):
        """test handling of hrtf loading failure."""
        with patch('py3dti.Core') as mock_core_class:
            mock_core = Mock()
            mock_core.LoadHRTF.return_value = False
            mock_core_class.return_value = mock_core
            
            with self.assertRaises(RuntimeError):
                AudioProcessor(hrtf_path="invalid_hrtf.sofa")
    
    def test_process_silent_input(self):
        """test processing silent input produces silent output."""
        session = Mock()
        session.listener_position = np.array([0.0, 0.0, 0.0])
        session.listener_forward = np.array([0.0, 0.0, -1.0])
        session.listener_up = np.array([0.0, 1.0, 0.0])
        session.sources = {}
        self.mock_core.ProcessBuffer.return_value = np.zeros((1024, 2), dtype=np.float32)
        
        output = self.processor.process_frame(session)
        
        self.assertEqual(output.shape, (1024, 2))
        self.assertTrue(np.allclose(output, 0.0))


class TestAudioStreamer(unittest.TestCase):
    """test cases for audiostreamer class."""
    
    def setUp(self):
        self.streamer = AudioStreamer()
    
    def test_encode_audio(self):
        """test audio encoding to base64."""
        audio = np.random.rand(1024, 2).astype(np.float32)
        
        encoded = self.streamer.encode_audio(audio)
        
        self.assertIsInstance(encoded, str)
        decoded_bytes = base64.b64decode(encoded)
        decoded_audio = np.frombuffer(decoded_bytes, dtype=np.float32).reshape(-1, 2)
        
        self.assertTrue(np.allclose(audio, decoded_audio))
    
    def test_decode_audio(self):
        """test audio decoding from base64."""
        original = np.random.rand(1024, 2).astype(np.float32)
        audio_bytes = original.tobytes()
        encoded = base64.b64encode(audio_bytes).decode('utf-8')
        
        decoded = self.streamer.decode_audio(encoded)
        
        self.assertTrue(np.allclose(original, decoded))
    
    def test_roundtrip_encoding(self):
        """test encode-decode roundtrip preserves data."""
        original = np.random.rand(512, 2).astype(np.float32)
        
        encoded = self.streamer.encode_audio(original)
        decoded = self.streamer.decode_audio(encoded)
        
        self.assertTrue(np.allclose(original, decoded))


class TestPerformanceMonitor(unittest.TestCase):
    """test cases for performancemonitor class."""
    
    def setUp(self):
        self.monitor = PerformanceMonitor(window_size=10)
    
    def test_record_latency(self):
        """test latency recording."""
        latencies = [10.0, 20.0, 15.0, 25.0]
        
        for latency in latencies:
            self.monitor.record_latency(latency)
        
        avg_latency = self.monitor.get_average_latency()
        self.assertEqual(avg_latency, np.mean(latencies))
    
    def test_record_processing_time(self):
        """test processing time recording."""
        times = [5.0, 10.0, 7.0, 12.0]
        
        for t in times:
            self.monitor.record_processing_time(t)
        
        avg_time = self.monitor.get_average_processing_time()
        self.assertEqual(avg_time, np.mean(times))
    
    def test_windowing(self):
        """test window size limiting."""
        for i in range(20):
            self.monitor.record_latency(float(i))
        avg = self.monitor.get_average_latency()
        expected = np.mean(range(10, 20))
        self.assertEqual(avg, expected)
    
    def test_get_statistics(self):
        """test statistics retrieval."""
        for i in range(10):
            self.monitor.record_latency(float(i))
            self.monitor.record_processing_time(float(i * 2))
        
        stats = self.monitor.get_statistics()
        
        required_keys = [
            'avg_latency', 'avg_processing_time',
            'min_latency', 'max_latency',
            'sample_count'
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
    
    def test_reset_statistics(self):
        """test statistics reset."""
        for i in range(5):
            self.monitor.record_latency(float(i))
        self.assertEqual(self.monitor.get_average_latency(), 2.0)
        self.monitor.reset()
        
        self.assertEqual(self.monitor.get_average_latency(), 0.0)


class TestIntegration(unittest.TestCase):
    """integration tests combining multiple components."""
    
    def setUp(self):
        self.session_manager = SessionManager(session_timeout=10)
        self.temp_dir = tempfile.mkdtemp()
        self.source_manager = SourceManager(upload_dir=self.temp_dir)
        self.monitor = PerformanceMonitor()
    
    def tearDown(self):
        self.session_manager.shutdown()
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_complete_workflow(self):
        """test complete workflow: create session, add source, process audio."""
        session_id = "integration_test"
        session = self.session_manager.create_session(session_id, 44100, 1024)
        
        self.assertIsNotNone(session)
        self.assertEqual(session.stream_state, StreamState.IDLE)
        source_id = self.source_manager.create_source(session_id)
        self.assertIsNotNone(source_id)
        test_tone = self.source_manager.generate_test_tone(440.0, 1.0, 44100)
        self.assertIsInstance(test_tone, np.ndarray)
        start_time = time.perf_counter()
        time.sleep(0.01)
        end_time = time.perf_counter()
        
        self.monitor.record_processing_time((end_time - start_time) * 1000)
        
        stats = self.monitor.get_statistics()
        self.assertGreater(stats['avg_processing_time'], 0)
    
    def test_multi_source_handling(self):
        """test handling multiple sources in one session."""
        session_id = "multi_source_test"
        session = self.session_manager.create_session(session_id)
        
        # create multiple sources
        source_ids = []
        for i in range(4):
            source_id = self.source_manager.create_source(session_id)
            source_ids.append(source_id)
        self.assertEqual(len(source_ids), 4)
        self.assertEqual(len(set(source_ids)), 4)
    
    def test_session_cleanup_integration(self):
        """test session cleanup removes associated resources."""
        session_id = "cleanup_test"
        session = self.session_manager.create_session(session_id)
        source_id = self.source_manager.create_source(session_id)
        session.last_seen = time.time() - 20
        expired = self.session_manager.cleanup_expired_sessions()
        
        self.assertIn(session_id, expired)
        self.assertIsNone(self.session_manager.get_session(session_id))


def suite():
    """create complete test suite."""
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSession))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSessionManager))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSourceManager))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAudioProcessor))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAudioStreamer))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPerformanceMonitor))
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())