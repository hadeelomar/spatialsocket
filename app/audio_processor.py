import numpy as np
from typing import Dict, Optional, Tuple

try:
    import py3dti
    PY3DTI_AVAILABLE = True
    print("py3dti library successfully loaded")
except ImportError:
    PY3DTI_AVAILABLE = False
    print("py3dti not available - using placeholder mode")

class AudioProcessor:

    def __init__(self, sample_rate: int = 48000, buffer_size: int = 1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.is_initialised = False

        self.renderer = None
        self.listener = None
        self.sources = {}
        print(f"AudioProcessor created: {sample_rate}Hz, {buffer_size} samples")

    def initialise(self, hrtf_file: str = 'p0200.sofa') -> bool:
        if not PY3DTI_AVAILABLE:
            print("py3dti not available - placeholder mode")
            self.is_initialised = True
            return True

        try:
            self.renderer = py3dti.BinauralRenderer(
                sample_rate = self.sample_rate,
                buffer_size = self.buffer_size
            )
            print("BinauralRenderer created")

            self.listener = self.renderer.create_listener()
            print("Listener created")

            hrtf_path = os.path.join('hrtf_datasets', hrtf_file)
            if os.path.exists(hrtf_path):
                success = self.listener.load_hrtf(hrtf_path)
                if success:
                    print(f"HRTF loaded: {hrtf_file}")
            
                    self.listener.enable_spatial_processing()
                    self.listener.enable_distance_attenuation()
                else:
                    print("Failed to load HRTF")
                    return False
            else:
                print(f"HRTF file not found: {hrtf_path}")
                return False

            self.is_initialised = True
            print("py3dti initialised successfully")
            return True
        
        except Exception as e:
            print(f"Failed to initialise py3dti: {e}")
            return False


    def create_source(self, source_id: str, position: Dict[str, float]) -> bool:
        if source_id in self.sources:
            return False
        
        if not PY3DTI_AVAILABLE or not self.is_initialised:
            self.sources[source_id] = {
                'position': position,
            }
            print(f"Created source (placeholder): {source_id}")
            return True

        try:
            source = self.renderer.create_source()
            source.enable_spatialisation()
            source.enable_distance_attenuation()
            source.set_position(position['x'], position['y'], position['z'])

            self.sources[source_id] = source
            print(f"Created source: {source_id} at {position}")
            return True
        
        except Exception as e:
            print(f"Failed to create source: {e}")
            return False
            

    def update_source_position(self, source_id: str, position: Dict[str, float]) -> bool:
        if source_id not in self.sources:
            return False
        
        if not PY3DTI_AVAILABLE:
            self.sources[source_id]['position'] = position
            return True
        
        try:
            source = self.sources[source_id]
            source.set_position(position['x'], position['y'], position['z'])
            return True
        
        except Exception as e:
            print(f"Failed to update position: {e}")
            return False

    def process_audio(self, source_id: str, audio_buffer: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if source_id not in self.sources or not self.is_initialised:
            return None
        
        if not PY3DTI_AVAILABLE:
            return audio_buffer.copy(), audio_buffer.copy()
        
        try:
            if len(audio_buffer) != self.buffer_size:
                if len(audio_buffer) < self.buffer_size:
                    audio_buffer = np.pad(audio_buffer, (0, self.buffer_size - len(audio_buffer)))
                else:
                    audio_buffer = audio_buffer[:self.buffer_size]
            
            if audio_buffer.dtype != np.float32:
                audio_buffer = audio_buffer.astype(np.float32)
                source = self.sources[source_id]
                source.set_buffer(audio_buffer)

                left = self.renderer.get_left_channel()
                right = self.renderer.get_right_channel()
                return left, right
        except Exception as e:
            print(f"Failed to process audio: {e}")

    def cleanup(self):
        self.sources.clear()
        self.is_initialised = False
        self.renderer = None
        self.listener = None
        print("AudioProcessor cleaned up")