import numpy as np
import base64
from typing import Optional


class AudioStreamer:
    """small compatibility wrapper around the module helpers."""

    def encode_audio(self, audio: np.ndarray) -> str:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2 or audio.shape[1] != 2:
            raise ValueError("expected stereo audio shaped (n, 2)")
        return encode_audio_to_base64(audio[:, 0], audio[:, 1])

    def decode_audio(self, encoded: str) -> np.ndarray:
        audio_bytes = base64.b64decode(encoded)
        return np.frombuffer(audio_bytes, dtype=np.float32).reshape(-1, 2)

def encode_audio_to_base64(left_channel: np.ndarray, right_channel: np.ndarray) -> str:
    """encode stereo audio to base64 for transmission."""
    # build interleaved stereo with a single allocation to reduce per-frame overhead.
    n = left_channel.shape[0]
    interleaved = np.empty(n * 2, dtype=np.float32)
    interleaved[0::2] = left_channel
    interleaved[1::2] = right_channel
    audio_bytes = interleaved.tobytes()
    encoded = base64.b64encode(audio_bytes).decode('utf-8')
    return encoded

def decode_audio_from_base64(encoded: str, buffer_size: int) -> Optional[np.ndarray]:
    """decode base64 audio data."""
    try:
        audio_bytes = base64.b64decode(encoded)
        audio = np.frombuffer(audio_bytes, dtype=np.float32)

        if len(audio) != buffer_size:
            print(f"Warning: expected {buffer_size} samples, got {len(audio)}")
            if len(audio) < buffer_size:
                audio = np.pad(audio, (0, buffer_size - len(audio)))
            else:
                audio = audio[:buffer_size]
        return audio
    
    except Exception as e:
        print(f"Error decoding audio: {e}")
        return None

def generate_test_tone(frequency: float, duration: float, sample_rate: int = 48000) -> np.ndarray:
    """generate a simple test tone."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint = False)
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)
