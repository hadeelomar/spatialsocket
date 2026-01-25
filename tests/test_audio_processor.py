#!/usr/bin/env python3
"""
Unit tests for AudioProcessor functionality
"""
import unittest
import numpy as np
import sys
import os

# Add the app directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

from audio_processor import AudioProcessor

class TestAudioProcessor(unittest.TestCase):
    """Test cases for AudioProcessor class"""
    
    def setUp(self):
        """Set up each test with a fresh AudioProcessor instance"""
        self.processor = AudioProcessor(sample_rate=48000, buffer_size=1024)
    
    def tearDown(self):
        """Clean up after each test"""
        if hasattr(self.processor, 'cleanup'):
            self.processor.cleanup()
    
    def test_01_creation(self):
        """Test AudioProcessor creation"""
        self.assertEqual(self.processor.sample_rate, 48000)
        self.assertEqual(self.processor.buffer_size, 1024)
        self.assertFalse(self.processor.is_initialised)
        self.assertFalse(self.processor.placeholder_mode)
        self.assertEqual(len(self.processor.sources), 0)
    
    def test_02_initialisation_placeholder_mode(self):
        """Test initialisation in placeholder mode (when py3dti not available)"""
        result = self.processor.initialise()
        
        self.assertTrue(result, "Initialisation should succeed in placeholder mode")
        self.assertTrue(self.processor.is_initialised, "Should be marked as initialised")
        self.assertTrue(self.processor.placeholder_mode, "Should be in placeholder mode")
    
    def test_03_source_creation_placeholder_mode(self):
        """Test source creation in placeholder mode"""
        # Initialise first
        self.processor.initialise()
        
        # Create source
        position = {'x': 1.0, 'y': 2.0, 'z': 3.0}
        result = self.processor.create_source('test_source', position)
        
        self.assertTrue(result, "Source creation should succeed")
        self.assertIn('test_source', self.processor.sources, "Source should be in sources dict")
        
        # Check stored position
        source_data = self.processor.sources['test_source']
        self.assertEqual(source_data['position'], position, "Position should be stored correctly")
    
    def test_04_duplicate_source_creation(self):
        """Test that creating duplicate sources fails"""
        self.processor.initialise()
        
        position = {'x': 0, 'y': 0, 'z': 0}
        
        # First creation should succeed
        result1 = self.processor.create_source('duplicate_test', position)
        self.assertTrue(result1, "First source creation should succeed")
        
        # Second creation should fail
        result2 = self.processor.create_source('duplicate_test', position)
        self.assertFalse(result2, "Duplicate source creation should fail")
    
    def test_05_source_position_update_placeholder_mode(self):
        """Test source position update in placeholder mode"""
        self.processor.initialise()
        
        # Create source
        initial_pos = {'x': 0, 'y': 0, 'z': 0}
        self.processor.create_source('update_test', initial_pos)
        
        # Update position
        new_pos = {'x': 5.0, 'y': -3.0, 'z': 2.0}
        result = self.processor.update_source_position('update_test', new_pos)
        
        self.assertTrue(result, "Position update should succeed")
        
        # Check updated position
        source_data = self.processor.sources['update_test']
        self.assertEqual(source_data['position'], new_pos, "Position should be updated")
    
    def test_06_position_update_nonexistent_source(self):
        """Test position update for non-existent source"""
        self.processor.initialise()
        
        result = self.processor.update_source_position('nonexistent', {'x': 1, 'y': 1, 'z': 1})
        self.assertFalse(result, "Position update for non-existent source should fail")
    
    def test_07_audio_processing_float32(self):
        """Test audio processing with float32 input"""
        self.processor.initialise()
        self.processor.create_source('audio_test', {'x': 0, 'y': 0, 'z': 0})
        
        # Create float32 audio data
        audio_data = np.random.randn(1024).astype(np.float32)
        
        result = self.processor.process_audio('audio_test', audio_data)
        
        self.assertIsNotNone(result, "Should return processing result")
        self.assertIsInstance(result, tuple, "Should return tuple")
        self.assertEqual(len(result), 2, "Should return stereo pair")
        
        left, right = result
        self.assertIsInstance(left, np.ndarray, "Left channel should be numpy array")
        self.assertIsInstance(right, np.ndarray, "Right channel should be numpy array")
        self.assertEqual(len(left), 1024, "Left channel should have correct length")
        self.assertEqual(len(right), 1024, "Right channel should have correct length")
    
    def test_08_audio_processing_int16_conversion(self):
        """Test audio processing with int16 input (should convert to float32)"""
        self.processor.initialise()
        self.processor.create_source('conversion_test', {'x': 0, 'y': 0, 'z': 0})
        
        # Create int16 audio data
        audio_data = (np.random.randn(1024) * 32767).astype(np.int16)
        
        result = self.processor.process_audio('conversion_test', audio_data)
        
        self.assertIsNotNone(result, "Should handle int16 conversion")
        self.assertIsInstance(result, tuple, "Should return tuple")
    
    def test_09_audio_processing_wrong_size_padding(self):
        """Test audio processing with undersized buffer (should pad)"""
        self.processor.initialise()
        self.processor.create_source('padding_test', {'x': 0, 'y': 0, 'z': 0})
        
        # Create smaller audio data
        audio_data = np.random.randn(500).astype(np.float32)
        
        result = self.processor.process_audio('padding_test', audio_data)
        
        self.assertIsNotNone(result, "Should handle undersized buffer")
        
        left, right = result
        self.assertEqual(len(left), 1024, "Should be padded to correct size")
        self.assertEqual(len(right), 1024, "Should be padded to correct size")
    
    def test_10_audio_processing_wrong_size_trimming(self):
        """Test audio processing with oversized buffer (should trim)"""
        self.processor.initialise()
        self.processor.create_source('trimming_test', {'x': 0, 'y': 0, 'z': 0})
        
        # Create larger audio data
        audio_data = np.random.randn(1500).astype(np.float32)
        
        result = self.processor.process_audio('trimming_test', audio_data)
        
        self.assertIsNotNone(result, "Should handle oversized buffer")
        
        left, right = result
        self.assertEqual(len(left), 1024, "Should be trimmed to correct size")
        self.assertEqual(len(right), 1024, "Should be trimmed to correct size")
    
    def test_11_audio_processing_2d_input_flattening(self):
        """Test audio processing with 2D input (should flatten)"""
        self.processor.initialise()
        self.processor.create_source('flatten_test', {'x': 0, 'y': 0, 'z': 0})
        
        # Create 2D audio data (single channel)
        audio_data = np.random.randn(1024, 1).astype(np.float32)
        
        result = self.processor.process_audio('flatten_test', audio_data)
        
        self.assertIsNotNone(result, "Should handle 2D input flattening")
        self.assertIsInstance(result, tuple, "Should return tuple")
    
    def test_12_audio_processing_invalid_shape(self):
        """Test audio processing with invalid shape (should fail)"""
        self.processor.initialise()
        self.processor.create_source('invalid_test', {'x': 0, 'y': 0, 'z': 0})
        
        # Create invalid 2D audio data (multiple channels)
        audio_data = np.random.randn(1024, 2).astype(np.float32)
        
        result = self.processor.process_audio('invalid_test', audio_data)
        
        self.assertIsNone(result, "Should reject invalid input shape")
    
    def test_13_audio_processing_nonexistent_source(self):
        """Test audio processing with non-existent source (should fail)"""
        self.processor.initialise()
        
        audio_data = np.random.randn(1024).astype(np.float32)
        
        result = self.processor.process_audio('nonexistent', audio_data)
        
        self.assertIsNone(result, "Should fail for non-existent source")
    
    def test_14_audio_processing_not_initialised(self):
        """Test audio processing without initialisation (should fail)"""
        # Don't initialise the processor
        self.processor.create_source('test', {'x': 0, 'y': 0, 'z': 0})
        
        audio_data = np.random.randn(1024).astype(np.float32)
        
        result = self.processor.process_audio('test', audio_data)
        
        self.assertIsNone(result, "Should fail when not initialised")
    
    def test_15_cleanup(self):
        """Test cleanup functionality"""
        # Initialise and create sources
        self.processor.initialise()
        self.processor.create_source('cleanup_test', {'x': 0, 'y': 0, 'z': 0})
        
        # Verify state before cleanup
        self.assertTrue(self.processor.is_initialised)
        self.assertEqual(len(self.processor.sources), 1)
        
        # Cleanup
        self.processor.cleanup()
        
        # Verify state after cleanup
        self.assertFalse(self.processor.is_initialised, "Should not be initialised after cleanup")
        self.assertFalse(self.processor.placeholder_mode, "Placeholder mode should be reset")
        self.assertEqual(len(self.processor.sources), 0, "Sources should be cleared")
        self.assertIsNone(self.processor.renderer, "Renderer should be None")
        self.assertIsNone(self.processor.listener, "Listener should be None")

class TestAudioProcessorEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_empty_audio_buffer(self):
        """Test processing with empty audio buffer"""
        processor = AudioProcessor(buffer_size=1024)
        processor.initialise()
        processor.create_source('empty_test', {'x': 0, 'y': 0, 'z': 0})
        
        empty_audio = np.array([]).astype(np.float32)
        result = processor.process_audio('empty_test', empty_audio)
        
        # Should pad empty buffer to correct size
        self.assertIsNotNone(result, "Should handle empty buffer")
        left, right = result
        self.assertEqual(len(left), 1024, "Should pad empty buffer to correct size")
    
    def test_extreme_values(self):
        """Test processing with extreme audio values"""
        processor = AudioProcessor(buffer_size=1024)
        processor.initialise()
        processor.create_source('extreme_test', {'x': 0, 'y': 0, 'z': 0})
        
        # Test with extreme values
        extreme_audio = np.array([1e10, -1e10] + [0] * 1022).astype(np.float32)
        result = processor.process_audio('extreme_test', extreme_audio)
        
        self.assertIsNotNone(result, "Should handle extreme values")

if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestAudioProcessor))
    suite.addTest(unittest.makeSuite(TestAudioProcessorEdgeCases))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
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
