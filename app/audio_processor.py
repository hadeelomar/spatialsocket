import numpy as np
import os
import time
from typing import Dict, Optional, Tuple

try:
    import py3dti
    PY3DTI_AVAILABLE = True
    print("py3dti library successfully loaded")
except ImportError:
    PY3DTI_AVAILABLE = False
    print("py3dti not available - using placeholder mode")

class AudioProcessor:
    """
    Audio processor for spatial audio processing.
    
    Supports mono audio input with the following expectations:
    - Input shape: (buffer_size,) for mono audio
    - Internal processing: float32 dtype
    - Output: Tuple of (left_channel, right_channel) as numpy arrays
    """

    def __init__(self, sample_rate: int = 48000, buffer_size: int = 1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.is_initialised = False
        self.placeholder_mode = False

        self.renderer = None
        self.listener = None
        self.sources = {}
        print(f"[AudioProcessor] Created: {sample_rate}Hz, {buffer_size} samples")

    def initialise(self, hrtf_file: str = 'p0200.sofa') -> bool:
        """
        Initialise the audio processor with HRTF file.
        
        Args:
            hrtf_file: Name of HRTF file in hrtf_datasets directory
            
        Returns:
            bool: True if initialisation successful, False otherwise
        """
        if not PY3DTI_AVAILABLE:
            print("[AudioProcessor] py3dti not available - enabling placeholder mode")
            self.placeholder_mode = True
            self.is_initialised = True
            return True

        try:
            print(f"[AudioProcessor] Initialising py3dti renderer...")
            self.renderer = py3dti.BinauralRenderer(
                sample_rate = self.sample_rate,
                buffer_size = self.buffer_size
            )
            print("[AudioProcessor] BinauralRenderer created")

            self.listener = self.renderer.create_listener()
            print("[AudioProcessor] Listener created")

            hrtf_path = os.path.join('hrtf_datasets', hrtf_file)
            print(f"[AudioProcessor] Loading HRTF from: {hrtf_path}")
            
            if os.path.exists(hrtf_path):
                success = self.listener.load_hrtf(hrtf_path)
                if success:
                    print(f"[AudioProcessor] HRTF loaded successfully: {hrtf_file}")
                    self.listener.enable_spatial_processing()
                    self.listener.enable_distance_attenuation()
                    print("[AudioProcessor] Spatial processing enabled")
                else:
                    print(f"[AudioProcessor] Failed to load HRTF: {hrtf_file}")
                    return False
            else:
                print(f"[AudioProcessor] HRTF file not found: {hrtf_path}")
                return False

            self.is_initialised = True
            self.placeholder_mode = False
            print("[AudioProcessor] py3dti initialised successfully")
            return True
        
        except Exception as e:
            print(f"[AudioProcessor] Failed to initialise py3dti: {e}")
            print("[AudioProcessor] Falling back to placeholder mode")
            self.placeholder_mode = True
            self.is_initialised = True
            return True


    def create_source(self, source_id: str, position: Dict[str, float]) -> bool:
        if source_id in self.sources:
            return False
        
        if self.placeholder_mode or not PY3DTI_AVAILABLE:
            self.sources[source_id] = {
                'position': position,
            }
            print(f"[AudioProcessor] Created source (placeholder): {source_id} at {position}")
            return True

        try:
            source = self.renderer.create_source()
            source.enable_spatialisation()
            source.enable_distance_attenuation()
            source.set_position(position['x'], position['y'], position['z'])

            self.sources[source_id] = source
            print(f"[AudioProcessor] Created source: {source_id} at {position}")
            return True
        
        except Exception as e:
            print(f"[AudioProcessor] Failed to create source: {e}")
            return False
            

    def update_source_position(self, source_id: str, position: Dict[str, float]) -> bool:
        if source_id not in self.sources:
            return False
        
        if self.placeholder_mode or not PY3DTI_AVAILABLE:
            self.sources[source_id]['position'] = position
            print(f"[AudioProcessor] Updated position (placeholder): {source_id} to {position}")
            return True
        
        try:
            source = self.sources[source_id]
            source.set_position(position['x'], position['y'], position['z'])
            return True
        
        except Exception as e:
            print(f"[AudioProcessor] Failed to update position: {e}")
            return False

    def set_listener_pose(self, position: Dict[str, float], orientation: Dict[str, Dict[str, float]], position_changed: bool = True) -> bool:
        """
        Set the listener position and orientation for spatial audio processing.
        
        Args:
            position: Dictionary with x, y, z coordinates
            orientation: Dictionary with forward and up vectors
            position_changed: Whether position actually changed (for optimisation)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_initialised:
            print("[AudioProcessor] Cannot set listener pose: processor not initialised")
            return False
        
        if self.placeholder_mode or not PY3DTI_AVAILABLE:
            # In placeholder mode, just store the pose for logging
            print(f"[AudioProcessor] Placeholder mode: set listener pose to {position}, orientation {orientation}")
            return True
        
        try:
            # Set listener position and orientation in py3dti
            self.listener.set_position(position['x'], position['y'], position['z'])
            
            # Set orientation using forward and up vectors
            forward = orientation['forward']
            up = orientation['up']
            
            # py3dti uses set_orientation with forward and up vectors
            self.listener.set_orientation(
                forward['x'], forward['y'], forward['z'],
                up['x'], up['y'], up['z']
            )
            
            # Log optimisation info
            if position_changed:
                print(f"[AudioProcessor] Listener position updated: {position}")
            else:
                print(f"[AudioProcessor] Listener orientation updated only (position unchanged)")
            
            return True
            
        except Exception as e:
            print(f"[AudioProcessor] Failed to set listener pose: {e}")
            return False

    def process_audio(self, source_id: str, audio_buffer: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Process audio buffer for spatial audio.
        
        Args:
            source_id: ID of the audio source
            audio_buffer: Mono audio data (shape: (buffer_size,))
            
        Returns:
            Tuple of (left_channel, right_channel) or None if processing fails
        """
        start_time = time.time()
        
        if source_id not in self.sources or not self.is_initialised:
            print(f"[AudioProcessor] Cannot process: source {source_id} not found or not initialised")
            return None
        
        if self.placeholder_mode:
            # Validate input shape even in placeholder mode
            if audio_buffer.ndim != 1:
                print(f"[AudioProcessor] Warning: Expected mono input (1D), got {audio_buffer.ndim}D")
                if audio_buffer.ndim == 2 and audio_buffer.shape[1] == 1:
                    audio_buffer = audio_buffer.flatten()
                else:
                    print(f"[AudioProcessor] Error: Unsupported input shape {audio_buffer.shape}")
                    return None
            
            # Ensure correct buffer size (same logic as full mode)
            if len(audio_buffer) != self.buffer_size:
                if len(audio_buffer) < self.buffer_size:
                    audio_buffer = np.pad(audio_buffer, (0, self.buffer_size - len(audio_buffer)))
                    print(f"[AudioProcessor] Padded audio from {len(audio_buffer) - (self.buffer_size - len(audio_buffer))} to {self.buffer_size} samples")
                else:
                    audio_buffer = audio_buffer[:self.buffer_size]
                    print(f"[AudioProcessor] Trimmed audio from {len(audio_buffer) + (len(audio_buffer) - self.buffer_size)} to {self.buffer_size} samples")
            
            # Convert to float32 if needed
            if audio_buffer.dtype != np.float32:
                original_dtype = audio_buffer.dtype
                audio_buffer = audio_buffer.astype(np.float32)
                print(f"[AudioProcessor] Converted audio from {original_dtype} to float32")
            
            # Placeholder mode: return identical left/right channels
            processed_audio = audio_buffer.copy()
            print(f"[AudioProcessor] Placeholder mode: processed {len(audio_buffer)} samples in {(time.time() - start_time)*1000:.2f}ms")
            return processed_audio, processed_audio
        
        try:
            # Validate and normalise input
            if audio_buffer.ndim != 1:
                print(f"[AudioProcessor] Warning: Expected mono input (1D), got {audio_buffer.ndim}D")
                if audio_buffer.ndim == 2 and audio_buffer.shape[1] == 1:
                    audio_buffer = audio_buffer.flatten()
                else:
                    print(f"[AudioProcessor] Error: Unsupported input shape {audio_buffer.shape}")
                    return None
            
            # Ensure correct buffer size
            if len(audio_buffer) != self.buffer_size:
                if len(audio_buffer) < self.buffer_size:
                    audio_buffer = np.pad(audio_buffer, (0, self.buffer_size - len(audio_buffer)))
                    print(f"[AudioProcessor] Padded audio from {len(audio_buffer) - (self.buffer_size - len(audio_buffer))} to {self.buffer_size} samples")
                else:
                    audio_buffer = audio_buffer[:self.buffer_size]
                    print(f"[AudioProcessor] Trimmed audio from {len(audio_buffer) + (len(audio_buffer) - self.buffer_size)} to {self.buffer_size} samples")
            
            # Convert to float32 if needed
            if audio_buffer.dtype != np.float32:
                original_dtype = audio_buffer.dtype
                audio_buffer = audio_buffer.astype(np.float32)
                print(f"[AudioProcessor] Converted audio from {original_dtype} to float32")
            
            # Set buffer for the source
            source = self.sources[source_id]
            source.set_buffer(audio_buffer)

            # Process spatial audio
            left = self.renderer.get_left_channel()
            right = self.renderer.get_right_channel()
            
            processing_time = (time.time() - start_time) * 1000
            print(f"[AudioProcessor] Processed frame for source {source_id} in {processing_time:.2f}ms")
            
            return left, right
            
        except Exception as e:
            print(f"[AudioProcessor] Failed to process audio: {e}")
            return None

    def cleanup(self):
        """Clean up audio processor resources."""
        print("[AudioProcessor] Cleaning up...")
        self.sources.clear()
        self.is_initialised = False
        self.placeholder_mode = False
        self.renderer = None
        self.listener = None
        print("[AudioProcessor] Cleanup complete")