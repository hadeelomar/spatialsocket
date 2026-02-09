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

    def __init__(self, sample_rate: int = 48000, buffer_size: int = 1024, max_sources: int = 16):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.max_sources = max_sources
        self.is_initialised = False
        self.placeholder_mode = False

        self.renderer = None
        self.listener = None
        self.environment = None
        self.sources = {}
        self.source_buffers = {}  # Store audio buffers for each source
        self.listener_pose = {'position': {'x': 0, 'y': 0, 'z': 0}}  # Store listener pose for placeholder mode
        print(f"[AudioProcessor] Created: {sample_rate}Hz, {buffer_size} samples, max_sources={max_sources}")

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

            # Add environment for room acoustics (optional, but improves realism)
            self.environment = self.renderer.add_environment()
            
            hrtf_path = os.path.join('hrtf_datasets', hrtf_file)
            print(f"[AudioProcessor] Loading HRTF from: {hrtf_path}")
            
            if os.path.exists(hrtf_path):
                success = self.listener.load_hrtf(hrtf_path)
                if success:
                    print(f"[AudioProcessor] HRTF loaded successfully: {hrtf_file}")
                    self.listener.enable_spatial_processing()
                    self.listener.enable_distance_attenuation()
                    
                    # Try to load BRIR for room acoustics if available
                    brir_path = os.path.join('hrtf_datasets', 'BRIR_medium.sofa')
                    if os.path.exists(brir_path):
                        env_success = self.environment.load_brir_from_sofa(brir_path)
                        if env_success:
                            print("[AudioProcessor] BRIR loaded for room acoustics")
                    
                    print("[AudioProcessor] Spatial processing enabled")
                else:
                    print(f"[AudioProcessor] Failed to load HRTF: {hrtf_file}")
                    return False
            else:
                print(f"[AudioProcessor] HRTF file not found: {hrtf_path}")
                return False

            self.is_initialised = True
            self.placeholder_mode = False
            print("[AudioProcessor] py3dti initialised successfully with full spatial capabilities")
            return True
        
        except Exception as e:
            print(f"[AudioProcessor] Failed to initialise py3dti: {e}")
            print("[AudioProcessor] Falling back to placeholder mode")
            self.placeholder_mode = True
            self.is_initialised = True
            return True


    def create_source(self, source_id: str, position: Dict[str, float]) -> bool:
        if source_id in self.sources:
            print(f"[AudioProcessor] Source {source_id} already exists")
            return False
        
        if len(self.sources) >= self.max_sources:
            print(f"[AudioProcessor] Maximum sources limit reached ({self.max_sources})")
            return False
        
        if self.placeholder_mode or not PY3DTI_AVAILABLE:
            self.sources[source_id] = {
                'position': position,
            }
            self.source_buffers[source_id] = np.zeros(self.buffer_size, dtype=np.float32)
            print(f"[AudioProcessor] Created source (placeholder): {source_id} at {position}")
            return True

        try:
            # Create py3dti source (following tutorial approach)
            source = self.renderer.create_source()
            source.set_position(position['x'], position['y'], position['z'])

            self.sources[source_id] = source
            self.source_buffers[source_id] = np.zeros(self.buffer_size, dtype=np.float32)
            
            print(f"[AudioProcessor] Created py3dti source: {source_id} at {position}")
            return True
        
        except Exception as e:
            print(f"[AudioProcessor] Failed to create source: {e}")
            return False
            

    def update_source_position(self, source_id: str, position: Dict[str, float]) -> bool:
        if source_id not in self.sources:
            print(f"[AudioProcessor] Source {source_id} not found for position update")
            return False
        
        if self.placeholder_mode or not PY3DTI_AVAILABLE:
            self.sources[source_id]['position'] = position
            print(f"[AudioProcessor] Updated position (placeholder): {source_id} to {position}")
            return True
        
        try:
            source = self.sources[source_id]
            source.set_position(position['x'], position['y'], position['z'])
            print(f"[AudioProcessor] Updated py3dti source position: {source_id} to {position}")
            return True
        
        except Exception as e:
            print(f"[AudioProcessor] Failed to update position: {e}")
            return False
    
    def remove_source(self, source_id: str) -> bool:
        """Remove a source and clean up its resources."""
        if source_id not in self.sources:
            print(f"[AudioProcessor] Source {source_id} not found for removal")
            return False
        
        try:
            if not self.placeholder_mode and PY3DTI_AVAILABLE:
                # Clean up py3dti source
                source = self.sources[source_id]
            
            # Remove from storage
            del self.sources[source_id]
            if source_id in self.source_buffers:
                del self.source_buffers[source_id]
            
            print(f"[AudioProcessor] Removed source: {source_id}")
            return True
            
        except Exception as e:
            print(f"[AudioProcessor] Failed to remove source: {e}")
            return False
    
    def get_source_count(self) -> int:
        """Get the current number of sources."""
        return len(self.sources)
    
    def get_source_ids(self) -> list:
        """Get list of current source IDs."""
        return list(self.sources.keys())

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
        # Store pose for placeholder mode calculations
        self.listener_pose['position'] = position.copy()
        self.listener_pose['orientation'] = {
            'forward': orientation['forward'].copy(),
            'up': orientation['up'].copy()
        }
        
        if not self.is_initialised:
            print("[AudioProcessor] Cannot set listener pose: processor not initialised")
            return False
        
        if self.placeholder_mode or not PY3DTI_AVAILABLE:
            # In placeholder mode, just store the pose for spatial calculations
            print(f"[AudioProcessor] Placeholder mode: set listener pose to {position}, orientation {orientation}")
            return True
        
        try:
            # Set listener position and orientation in py3dti
            self.listener.set_position(position['x'], position['y'], position['z'])
            
            # Set orientation using forward and up vectors
            forward = orientation['forward']
            up = orientation['up']
            
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
        
        # Store the audio buffer for mixing
        self.source_buffers[source_id] = audio_buffer.copy()
        
        # Mix all active sources
        return self._mix_all_sources()
    
    def _mix_all_sources(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Mix all active sources and return stereo output."""
        if not self.sources:
            return None
        
        start_time = time.time()
        
        if self.placeholder_mode:
            return self._mix_sources_placeholder()
        
        try:
            # Set buffers for all sources
            for source_id, source in self.sources.items():
                if source_id in self.source_buffers:
                    buffer = self.source_buffers[source_id]
                    if buffer is not None and len(buffer) == self.buffer_size:
                        source.set_buffer(buffer)

            left = self.renderer.get_left_channel()
            right = self.renderer.get_right_channel()
            
            processing_time = (time.time() - start_time) * 1000
            print(f"[AudioProcessor] Rendered {len(self.sources)} py3dti sources in {processing_time:.2f}ms")
            
            return left, right
            
        except Exception as e:
            print(f"[AudioProcessor] Failed to mix audio: {e}")
            return None
    
    def _mix_sources_placeholder(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enhanced placeholder mode that simulates spatial audio effects.
        Provides basic stereo panning and distance attenuation when py3dti is not available.
        """
        if not self.source_buffers:
            return np.zeros(self.buffer_size, dtype=np.float32), np.zeros(self.buffer_size, dtype=np.float32)
        
        mixed_left = np.zeros(self.buffer_size, dtype=np.float32)
        mixed_right = np.zeros(self.buffer_size, dtype=np.float32)
        
        # Get listener position for spatial calculations
        listener_pos = self.listener_pose['position']
        
        for source_id, buffer in self.source_buffers.items():
            if buffer is None or len(buffer) != self.buffer_size:
                continue
                
            # Get source position (stored differently in placeholder mode)
            source_pos = {'x': 0, 'y': 0, 'z': 0}
            if isinstance(self.sources.get(source_id), dict):
                source_pos = self.sources[source_id].get('position', {'x': 0, 'y': 0, 'z': 0})
            
            # Calculate distance-based attenuation
            distance = np.sqrt(
                (source_pos['x'] - listener_pos['x'])**2 +
                (source_pos['y'] - listener_pos['y'])**2 +
                (source_pos['z'] - listener_pos['z'])**2
            )
            
            # Simple distance attenuation (inverse square law)
            attenuation = 1.0 / (1.0 + distance * 0.1)
            attenuation = np.clip(attenuation, 0.1, 1.0)
            
            # Calculate stereo panning based on source position
            # Simple left-right panning based on x position
            pan = np.clip((source_pos['x'] - listener_pos['x']) / 5.0, -1.0, 1.0)
            
            # Apply attenuation and panning
            attenuated_buffer = buffer * attenuation
            
            # Pan to left and right channels
            left_gain = np.sqrt(1.0 - pan) * 0.707  # -3dB compensation
            right_gain = np.sqrt(1.0 + pan) * 0.707
            
            mixed_left += attenuated_buffer * left_gain
            mixed_right += attenuated_buffer * right_gain
        
        # Normalise to prevent clipping
        max_val = np.max(np.abs(np.concatenate([mixed_left, mixed_right])))
        if max_val > 0.95:
            scale_factor = 0.95 / max_val
            mixed_left *= scale_factor
            mixed_right *= scale_factor
        
        processing_time = (time.time() - start_time) * 1000
        print(f"[AudioProcessor] Mixed {len(self.source_buffers)} sources with spatial simulation in {processing_time:.2f}ms")
        
        return mixed_left, mixed_right

    def cleanup(self):
        """Clean up audio processor resources."""
        print("[AudioProcessor] Cleaning up...")
        self.sources.clear()
        self.source_buffers.clear()
        self.is_initialised = False
        self.placeholder_mode = False
        self.renderer = None
        self.listener = None
        self.environment = None
        print("[AudioProcessor] Cleanup complete")