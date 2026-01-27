"""
Unit tests for listener pose functionality
"""

import unittest
import time
import threading
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from session_manager import SessionManager, Session, StreamState
from audio_processor import AudioProcessor
from main import _validate_listener_payload


class TestListenerValidation(unittest.TestCase):
    """Test cases for listener payload validation"""
    
    def test_valid_payload(self):
        """Test validation of correct payload"""
        valid_payload = {
            'position': {'x': 1.0, 'y': 2.0, 'z': 3.0},
            'orientation': {
                'forward': {'x': 0.0, 'y': 0.0, 'z': -1.0},
                'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
            }
        }
        
        error = _validate_listener_payload(valid_payload)
        self.assertIsNone(error)
    
    def test_missing_required_keys(self):
        """Test validation fails with missing keys"""
        # Missing position
        payload = {'orientation': {'forward': {'x': 0, 'y': 0, 'z': -1}, 'up': {'x': 0, 'y': 1, 'z': 0}}}
        error = _validate_listener_payload(payload)
        self.assertIsNotNone(error)
        self.assertIn('Missing required key: position', error)
        
        # Missing orientation
        payload = {'position': {'x': 1, 'y': 2, 'z': 3}}
        error = _validate_listener_payload(payload)
        self.assertIsNotNone(error)
        self.assertIn('Missing required key: orientation', error)
    
    def test_invalid_position_types(self):
        """Test validation fails with wrong position types"""
        payload = {
            'position': 'not_a_dict',
            'orientation': {'forward': {'x': 0, 'y': 0, 'z': -1}, 'up': {'x': 0, 'y': 1, 'z': 0}}
        }
        error = _validate_listener_payload(payload)
        self.assertIsNotNone(error)
        self.assertIn('Position must be a dictionary', error)
    
    def test_missing_position_coordinates(self):
        """Test validation fails with missing position coordinates"""
        payload = {
            'position': {'x': 1, 'y': 2},  # Missing z
            'orientation': {'forward': {'x': 0, 'y': 0, 'z': -1}, 'up': {'x': 0, 'y': 1, 'z': 0}}
        }
        error = _validate_listener_payload(payload)
        self.assertIsNotNone(error)
        self.assertIn('Missing required position key: z', error)
    
    def test_invalid_position_ranges(self):
        """Test validation fails with position values outside reasonable range"""
        payload = {
            'position': {'x': 2000, 'y': 0, 'z': 0},  # x is too large
            'orientation': {'forward': {'x': 0, 'y': 0, 'z': -1}, 'up': {'x': 0, 'y': 1, 'z': 0}}
        }
        error = _validate_listener_payload(payload)
        self.assertIsNotNone(error)
        self.assertIn('is outside reasonable range', error)
    
    def test_invalid_orientation_vectors(self):
        """Test validation fails with invalid orientation vectors"""
        payload = {
            'position': {'x': 0, 'y': 0, 'z': 0},
            'orientation': {
                'forward': {'x': 0, 'y': 0, 'z': -1},
                'up': 'not_a_dict'
            }
        }
        error = _validate_listener_payload(payload)
        self.assertIsNotNone(error)
        self.assertIn('Orientation up must be a dictionary', error)
    
    def test_parallel_vectors(self):
        """Test validation fails with parallel forward and up vectors"""
        payload = {
            'position': {'x': 0, 'y': 0, 'z': 0},
            'orientation': {
                'forward': {'x': 0, 'y': 1, 'z': 0},  # Both pointing up
                'up': {'x': 0, 'y': 1, 'z': 0}
            }
        }
        error = _validate_listener_payload(payload)
        self.assertIsNotNone(error)
        self.assertIn('should not be parallel', error)


class TestSessionListenerPose(unittest.TestCase):
    """Test cases for session listener pose management"""
    
    def setUp(self):
        self.session = Session("test_session", 48000, 1024)
    
    def test_initial_listener_pose(self):
        """Test initial listener pose is set correctly"""
        pose = self.session.get_listener_pose()
        
        self.assertIn('position', pose)
        self.assertIn('orientation', pose)
        self.assertEqual(pose['position'], {'x': 0.0, 'y': 0.0, 'z': 0.0})
        self.assertEqual(pose['orientation']['forward'], {'x': 0.0, 'y': 0.0, 'z': -1.0})
        self.assertEqual(pose['orientation']['up'], {'x': 0.0, 'y': 1.0, 'z': 0.0})
        self.assertTrue(self.session.listener_position_changed)  # Initial state
    
    def test_update_listener_pose_position_change(self):
        """Test updating listener pose with position change"""
        new_position = {'x': 5.0, 'y': 2.0, 'z': -3.0}
        new_orientation = {
            'forward': {'x': 1.0, 'y': 0.0, 'z': 0.0},
            'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
        }
        
        position_changed = self.session.update_listener_pose(new_position, new_orientation)
        
        self.assertTrue(position_changed)
        self.assertTrue(self.session.listener_position_changed)
        
        pose = self.session.get_listener_pose()
        self.assertEqual(pose['position'], new_position)
        self.assertEqual(pose['orientation'], new_orientation)
    
    def test_update_listener_pose_orientation_only(self):
        """Test updating listener pose with only orientation change"""
        # First update to establish baseline
        initial_position = {'x': 5.0, 'y': 2.0, 'z': -3.0}
        initial_orientation = {
            'forward': {'x': 1.0, 'y': 0.0, 'z': 0.0},
            'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
        }
        self.session.update_listener_pose(initial_position, initial_orientation)
        
        # Update with same position, different orientation
        new_orientation = {
            'forward': {'x': 0.0, 'y': 0.0, 'z': -1.0},
            'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
        }
        
        position_changed = self.session.update_listener_pose(initial_position, new_orientation)
        
        self.assertFalse(position_changed)  # Position didn't change
        self.assertFalse(self.session.listener_position_changed)
        
        pose = self.session.get_listener_pose()
        self.assertEqual(pose['position'], initial_position)  # Position unchanged
        self.assertEqual(pose['orientation'], new_orientation)  # Orientation updated
    
    def test_update_listener_pose_small_position_change(self):
        """Test optimisation ignores very small position changes"""
        initial_position = {'x': 5.0, 'y': 2.0, 'z': -3.0}
        new_position = {'x': 5.0005, 'y': 2.0005, 'z': -3.0005}  # Very small change
        orientation = {
            'forward': {'x': 1.0, 'y': 0.0, 'z': 0.0},
            'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
        }
        
        position_changed = self.session.update_listener_pose(new_position, orientation)
        
        self.assertFalse(position_changed)  # Change too small to matter
        self.assertFalse(self.session.listener_position_changed)


class TestAudioProcessorListenerPose(unittest.TestCase):
    """Test cases for audio processor listener pose functionality"""
    
    def setUp(self):
        self.processor = AudioProcessor(48000, 1024)
        # Initialise in placeholder mode for testing
        self.processor.placeholder_mode = True
        self.processor.is_initialised = True
    
    def tearDown(self):
        self.processor.cleanup()
    
    def test_set_listener_pose_placeholder_mode(self):
        """Test setting listener pose in placeholder mode"""
        position = {'x': 1.0, 'y': 2.0, 'z': 3.0}
        orientation = {
            'forward': {'x': 0.0, 'y': 0.0, 'z': -1.0},
            'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
        }
        
        result = self.processor.set_listener_pose(position, orientation, True)
        
        self.assertTrue(result)
    
    def test_set_listener_pose_not_initialised(self):
        """Test setting listener pose when not initialised"""
        self.processor.is_initialised = False
        
        position = {'x': 1.0, 'y': 2.0, 'z': 3.0}
        orientation = {
            'forward': {'x': 0.0, 'y': 0.0, 'z': -1.0},
            'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
        }
        
        result = self.processor.set_listener_pose(position, orientation, True)
        
        self.assertFalse(result)
    
    @patch('spatialsocket.app.audio_processor.PY3DTI_AVAILABLE', True)
    def test_set_listener_pose_with_py3dti(self):
        """Test setting listener pose with py3dti (mocked)"""
        # Mock py3dti components
        mock_listener = MagicMock()
        mock_listener.set_position = MagicMock()
        mock_listener.set_orientation = MagicMock()
        
        self.processor.placeholder_mode = False
        self.processor.listener = mock_listener
        
        position = {'x': 1.0, 'y': 2.0, 'z': 3.0}
        orientation = {
            'forward': {'x': 0.0, 'y': 0.0, 'z': -1.0},
            'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
        }
        
        result = self.processor.set_listener_pose(position, orientation, True)
        
        self.assertTrue(result)
        mock_listener.set_position.assert_called_once_with(1.0, 2.0, 3.0)
        mock_listener.set_orientation.assert_called_once_with(0.0, 0.0, -1.0, 0.0, 1.0, 0.0)


class TestListenerIntegration(unittest.TestCase):
    """Integration tests for listener functionality"""
    
    def setUp(self):
        self.session_manager = SessionManager(session_timeout=5, cleanup_interval=1)
        self.session = self.session_manager.create_session("test_session")
        self.processor = AudioProcessor(48000, 1024)
        self.processor.placeholder_mode = True
        self.processor.is_initialised = True
    
    def tearDown(self):
        self.processor.cleanup()
        self.session_manager.shutdown()
    
    def test_full_listener_update_workflow(self):
        """Test complete listener update workflow"""
        # Simulate listener update data
        position = {'x': 10.0, 'y': 5.0, 'z': -2.0}
        orientation = {
            'forward': {'x': 0.707, 'y': 0.0, 'z': -0.707},  # 45-degree rotation
            'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
        }
        
        # Update session
        position_changed = self.session.update_listener_pose(position, orientation)
        self.assertTrue(position_changed)
        
        # Update audio processor
        result = self.processor.set_listener_pose(position, orientation, position_changed)
        self.assertTrue(result)
        
        # Verify session state
        pose = self.session.get_listener_pose()
        self.assertEqual(pose['position'], position)
        self.assertEqual(pose['orientation'], orientation)
    
    def test_high_frequency_updates(self):
        """Test handling of high-frequency listener updates"""
        base_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        base_orientation = {
            'forward': {'x': 0.0, 'y': 0.0, 'z': -1.0},
            'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
        }
        
        # Simulate 60 updates (1 second at 60Hz)
        updates_processed = 0
        position_changes = 0
        
        for i in range(60):
            # Small position change every 10 updates
            if i % 10 == 0:
                position = {'x': i * 0.1, 'y': 0.0, 'z': 0.0}
            else:
                position = base_position
            
            # Small orientation change each update
            angle = (i * 0.1) % 360
            import math
            rad = math.radians(angle)
            orientation = {
                'forward': {'x': math.sin(rad), 'y': 0.0, 'z': -math.cos(rad)},
                'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
            }
            
            position_changed = self.session.update_listener_pose(position, orientation)
            if position_changed:
                position_changes += 1
            
            result = self.processor.set_listener_pose(position, orientation, position_changed)
            if result:
                updates_processed += 1
        
        # Verify optimisation worked (should have fewer position changes than total updates)
        self.assertLess(position_changes, updates_processed)
        self.assertEqual(position_changes, 6)  # Position changed every 10 updates
        self.assertEqual(updates_processed, 60)  # All updates processed
    
    def test_concurrent_listener_updates(self):
        """Test thread safety of concurrent listener updates"""
        def update_listener_thread(thread_id):
            for i in range(10):
                position = {'x': thread_id * 1.0, 'y': i * 0.1, 'z': 0.0}
                orientation = {
                    'forward': {'x': 0.0, 'y': 0.0, 'z': -1.0},
                    'up': {'x': 0.0, 'y': 1.0, 'z': 0.0}
                }
                
                position_changed = self.session.update_listener_pose(position, orientation)
                result = self.processor.set_listener_pose(position, orientation, position_changed)
                self.assertTrue(result)
        
        # Create multiple threads updating listener simultaneously
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_listener_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify final state is consistent
        pose = self.session.get_listener_pose()
        self.assertIsInstance(pose['position']['x'], (int, float))
        self.assertIsInstance(pose['position']['y'], (int, float))
        self.assertIsInstance(pose['position']['z'], (int, float))


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during tests
    
    # Run tests
    unittest.main(verbosity=2)
