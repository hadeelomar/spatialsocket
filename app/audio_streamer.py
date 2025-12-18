import numpy as np
import base64
from typing import Optional, Tuple

def encode_audio_to_base64(left_channel: np.ndarray, right_channel: np.ndarray) -> str:
    """Encodes stereo audio to base64 for transmission"""
    stereo = np.vstack([left_channel, right_channel]).T.flatten()
    audio_bytes = stereo.astype(np.float32).tobytes()
    encoded = base64.b64encode(audio_bytes).decode('utf-8')
    return encoded

def decode_audio_from_base64(encoded: str, buffer_size: int) -> Optional[np.ndarray]:
    try:
        audio_bytes = base64.b64decode(encoded)
        audio = np.frombuffer(audio_bytes, dtype=np.float32)

        if len(audio) != buffer_size:
            print(f"Warning: expected {buffer_size} samples, instead got {len(audio)}")
            if len(audio) < buffer_size:
                audio = np.pad(audio, (0, buffer_size - len(audio)))
            else:
                audio = audio[:buffer_size]
        return audio
    
    except Exception as e:
        print(f"Error decoding audio: {e}")
        return None

def generate_test_tone(frequency: float, duration: float, sample_rate: int = 48000) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint = False)
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)