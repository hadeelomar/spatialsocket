"""
Unit tests for multiple sources functionality
"""
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from audio_processor import AudioProcessor
from performance_monitor import PerformanceMonitor
from session_manager import SessionManager, Session

class TestMultipleSources(unittest.TestCase):
    """Test cases for multiple sources functionality"""
    
    def setUp(self):
        """Set up each test with fresh instances"""
        self.processor = AudioProcessor(sample_rate=48000, buffer_size=1024, max_sources=4)
        self.monitor = PerformanceMonitor()
        self.session_manager = SessionManager()
        
        # Initialise processor in placeholder mode for testing
        self.processor.initialise()
    
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self.processor, 'cleanup'):
            self.processor.cleanup()
    
    def test_01_processor_creation_with_limits(self):
        """Test AudioProcessor creation with source limits"""
        self.assertEqual(self.processor.max_sources, 4)
        self.assertEqual(self.processor.sample_rate, 48000)
        self.assertEqual(self.processor.buffer_size, 1024)
        self.assertFalse(self.processor.is_initialised)
        self.assertEqual(len(self.processor.sources), 0)
        self.assertEqual(len(self.processor.source_buffers), 0)
    
    def test_02_multiple_source_creation(self):
        """Test creating multiple sources up to the limit"""
        positions = [
            {'x': -2, 'y': 0, 'z': 0},
            {'x': 2, 'y': 0, 'z': 0},
            {'x': 0, 'y': 1, 'z': -2},
            {'x': 1, 'y': -1, 'z': 1}
        ]
        
        # Create sources up to the limit
        for i, position in enumerate(positions):
            source_id = f'source_{i}'
            result = self.processor.create_source(source_id, position)
            self.assertTrue(result, f"Should create source {source_id}")
        
        self.assertEqual(self.processor.get_source_count(), 4)
        self.assertEqual(set(self.processor.get_source_ids()), 
                        {'source_0', 'source_1', 'source_2', 'source_3'})
    
    def test_03_source_limit_enforcement(self):
        """Test that source limits are enforced"""
        # Create sources up to the limit
        for i in range(4):
            source_id = f'source_{i}'
            result = self.processor.create_source(source_id, {'x': i, 'y': 0, 'z': 0})
            self.assertTrue(result, f"Should create source {source_id}")
        
        # Try to create one more (should fail)
        result = self.processor.create_source('source_extra', {'x': 10, 'y': 0, 'z': 0})
        self.assertFalse(result, "Should not create source beyond limit")
        self.assertEqual(self.processor.get_source_count(), 4)
    
    def test_04_duplicate_source_creation(self):
        """Test that duplicate source creation is prevented"""
        position = {'x': 1, 'y': 2, 'z': 3}
        
        # Create source
        result1 = self.processor.create_source('duplicate_source', position)
        self.assertTrue(result1, "First creation should succeed")
        
        # Try to create same source again
        result2 = self.processor.create_source('duplicate_source', position)
        self.assertFalse(result2, "Duplicate creation should fail")
        self.assertEqual(self.processor.get_source_count(), 1)
    
    def test_05_independent_source_positions(self):
        """Test that sources maintain independent positions"""
        # Create multiple sources at different positions
        positions = {
            'source_left': {'x': -3, 'y': 0, 'z': 0},
            'source_right': {'x': 3, 'y': 0, 'z': 0},
            'source_above': {'x': 0, 'y': 2, 'z': -1}
        }
        
        for source_id, position in positions.items():
            result = self.processor.create_source(source_id, position)
            self.assertTrue(result, f"Should create {source_id}")
        
        # Update positions independently
        new_positions = {
            'source_left': {'x': -4, 'y': 1, 'z': 0},
            'source_right': {'x': 4, 'y': -1, 'z': 1},
            'source_above': {'x': 0, 'y': 3, 'z': -2}
        }
        
        for source_id, position in new_positions.items():
            result = self.processor.update_source_position(source_id, position)
            self.assertTrue(result, f"Should update position for {source_id}")
        
        # Verify all sources still exist
        self.assertEqual(self.processor.get_source_count(), 3)
    
    def test_06_source_removal(self):
        """Test source removal and cleanup"""
        # Create multiple sources
        source_ids = ['source_1', 'source_2', 'source_3']
        for source_id in source_ids:
            self.processor.create_source(source_id, {'x': 0, 'y': 0, 'z': 0})
        
        self.assertEqual(self.processor.get_source_count(), 3)
        
        # Remove one source
        result = self.processor.remove_source('source_2')
        self.assertTrue(result, "Should remove source_2")
        self.assertEqual(self.processor.get_source_count(), 2)
        self.assertNotIn('source_2', self.processor.get_source_ids())
        self.assertNotIn('source_2', self.processor.source_buffers)
        
        # Remove another source
        result = self.processor.remove_source('source_1')
        self.assertTrue(result, "Should remove source_1")
        self.assertEqual(self.processor.get_source_count(), 1)
        self.assertEqual(self.processor.get_source_ids(), ['source_3'])
    
    def test_07_remove_nonexistent_source(self):
        """Test removing a source that doesn't exist"""
        result = self.processor.remove_source('nonexistent_source')
        self.assertFalse(result, "Should not remove nonexistent source")
    
    def test_08_audio_mixing_multiple_sources(self):
        """Test that audio from multiple sources is mixed correctly"""
        # Create two sources
        self.processor.create_source('source_a', {'x': -1, 'y': 0, 'z': 0})
        self.processor.create_source('source_b', {'x': 1, 'y': 0, 'z': 0})
        
        # Generate different test audio for each source
        audio_a = 0.3 * np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 4800))  # 440Hz
        audio_b = 0.3 * np.sin(2 * np.pi * 554 * np.linspace(0, 0.1, 4800))  # 554Hz
        
        # Process audio from source_a
        result_a = self.processor.process_audio('source_a', audio_a[:1024])
        self.assertIsNotNone(result_a, "Should process audio from source_a")
        left_a, right_a = result_a
        
        # Process audio from source_b
        result_b = self.processor.process_audio('source_b', audio_b[:1024])
        self.assertIsNotNone(result_b, "Should process audio from source_b")
        left_b, right_b = result_b
        
        # In placeholder mode, both should return the mixed audio
        # The exact mixing depends on implementation, but should not be identical
        # to a single source's input (unless both sources have same audio)
        self.assertEqual(left_a.shape, (1024,))
        self.assertEqual(right_a.shape, (1024,))
        self.assertEqual(left_b.shape, (1024,))
        self.assertEqual(right_b.shape, (1024,))
    
    def test_09_audio_processing_after_source_removal(self):
        """Test audio processing after source removal"""
        # Create two sources
        self.processor.create_source('source_keep', {'x': 0, 'y': 0, 'z': 0})
        self.processor.create_source('source_remove', {'x': 1, 'y': 0, 'z': 0})
        
        # Process audio from both sources
        audio = np.random.randn(1024).astype(np.float32) * 0.1
        result1 = self.processor.process_audio('source_keep', audio)
        result2 = self.processor.process_audio('source_remove', audio)
        
        self.assertIsNotNone(result1, "Should process audio from source_keep")
        self.assertIsNotNone(result2, "Should process audio from source_remove")
        
        # Remove one source
        self.processor.remove_source('source_remove')
        
        # Try to process audio from removed source (should fail)
        result_removed = self.processor.process_audio('source_remove', audio)
        self.assertIsNone(result_removed, "Should not process audio from removed source")
        
        # Should still process audio from remaining source
        result_keep = self.processor.process_audio('source_keep', audio)
        self.assertIsNotNone(result_keep, "Should still process audio from remaining source")
    
    def test_10_performance_monitor_source_counting(self):
        """Test performance monitor tracks source counts correctly"""
        session_id = 'test_session'
        
        # Simulate session activity
        self.monitor.record_receive_timestamp(session_id)
        
        # Create sources and update metrics
        for i in range(3):
            self.monitor.increment_source_count(session_id)
        
        # Check metrics
        metrics = self.monitor.to_dict()
        session_metrics = metrics['sessions'][session_id]
        self.assertEqual(session_metrics['source_count'], 3)
        
        # Remove sources and update metrics
        self.monitor.decrement_source_count(session_id)
        self.monitor.decrement_source_count(session_id)
        
        metrics = self.monitor.to_dict()
        session_metrics = metrics['sessions'][session_id]
        self.assertEqual(session_metrics['source_count'], 1)
        
        # Test decrement below zero
        self.monitor.decrement_source_count(session_id)
        self.monitor.decrement_source_count(session_id)  # Should go to 0, not negative
        
        metrics = self.monitor.to_dict()
        session_metrics = metrics['sessions'][session_id]
        self.assertEqual(session_metrics['source_count'], 0)
    
    def test_11_rapid_create_remove_cycles(self):
        """Test stability under rapid create/remove cycles"""
        cycles = 20
        success_count = 0
        
        for i in range(cycles):
            source_id = f'temp_source_{i}'
            
            # Create source
            if self.processor.create_source(source_id, {'x': i*0.1, 'y': 0, 'z': 0}):
                # Update position
                if self.processor.update_source_position(source_id, {'x': i*0.1, 'y': 1, 'z': 0}):
                    # Remove source
                    if self.processor.remove_source(source_id):
                        success_count += 1
        
        success_rate = (success_count / cycles) * 100
        self.assertGreaterEqual(success_rate, 90, f"Success rate should be >= 90%, got {success_rate}%")
        self.assertEqual(self.processor.get_source_count(), 0, "All sources should be cleaned up")
    
    def test_12_source_buffer_cleanup(self):
        """Test that source buffers are properly cleaned up"""
        # Create source and process audio
        self.processor.create_source('test_buffer', {'x': 0, 'y': 0, 'z': 0})
        audio = np.random.randn(1024).astype(np.float32) * 0.1
        self.processor.process_audio('test_buffer', audio)
        
        # Verify buffer exists
        self.assertIn('test_buffer', self.processor.source_buffers)
        self.assertEqual(self.processor.source_buffers['test_buffer'].shape, (1024,))
        
        # Remove source
        self.processor.remove_source('test_buffer')
        
        # Verify buffer is cleaned up
        self.assertNotIn('test_buffer', self.processor.source_buffers)
    
    def test_13_maximum_sources_configuration(self):
        """Test different maximum sources configurations"""
        # Test with different limits
        for max_sources in [1, 2, 8, 16]:
            processor = AudioProcessor(max_sources=max_sources)
            self.assertEqual(processor.max_sources, max_sources)
            
            # Create sources up to the limit
            for i in range(max_sources):
                result = processor.create_source(f'source_{i}', {'x': i, 'y': 0, 'z': 0})
                self.assertTrue(result, f"Should create source {i} with limit {max_sources}")
            
            # Try to create one more
            result = processor.create_source('extra', {'x': 99, 'y': 0, 'z': 0})
            self.assertFalse(result, f"Should not exceed limit {max_sources}")
            
            processor.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
