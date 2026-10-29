"""source manager for audio files and conversion."""
import os
import numpy as np
import base64
import logging
import uuid
from typing import Optional

try:
    import py3dti
    from miniaudio import decode_file, SampleFormat
    PY3DTI_AVAILABLE = True
    print("py3dti and miniaudio libraries successfully loaded")
except ImportError:
    PY3DTI_AVAILABLE = False
    print("py3dti/miniaudio not available - using basic audio processing")


class SourceManager:
    """manages audio files and converts them for processing."""

    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        self.temp_files = {}
        os.makedirs(upload_dir, exist_ok=True)
        logging.info(f"sourcemanager initialized with upload_dir: {upload_dir}")

    def save_uploaded_file(
        self,
        session_id: str,
        source_id: str,
        file_data: bytes,
        filename: str,
    ) -> Optional[str]:
        """save raw bytes to disk."""
        try:
            session_dir = os.path.join(self.upload_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            safe_filename = f"{source_id}_{filename}"
            file_path = os.path.join(session_dir, safe_filename)
            with open(file_path, "wb") as f:
                f.write(file_data)
            if session_id not in self.temp_files:
                self.temp_files[session_id] = []
            self.temp_files[session_id].append(file_path)
            logging.info(f"Saved uploaded file: {file_path}")
            return file_path
        except Exception as e:
            logging.error(f"Failed to save uploaded file: {e}")
            return None

    def create_source(self, session_id: str) -> str:
        """generate a unique source id for a session."""
        return f"{session_id}_{uuid.uuid4().hex[:8]}"

    def generate_test_tone(
        self,
        frequency: float,
        duration: float,
        sample_rate: int = 44100,
    ) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return (0.3 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    def _stereo_to_mono(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            return audio
        return audio.mean(axis=1, dtype=np.float32)

    def _split_into_buffers(self, audio_data: np.ndarray, buffer_size: int) -> list:
        audio_data = np.asarray(audio_data, dtype=np.float32)
        remainder = len(audio_data) % buffer_size
        if remainder:
            audio_data = np.pad(audio_data, (0, buffer_size - remainder))
        return [audio_data[i:i + buffer_size] for i in range(0, len(audio_data), buffer_size)]

    def _resample_audio(
        self,
        audio_data: np.ndarray,
        original_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        audio_data = np.asarray(audio_data, dtype=np.float32)
        if original_rate == target_rate:
            return audio_data
        duration = len(audio_data) / float(original_rate)
        target_length = int(round(duration * target_rate))
        if target_length <= 0:
            return np.zeros(0, dtype=np.float32)
        # use simple interpolation here because this helper is mainly for tests and compatibility paths.
        source_x = np.linspace(0.0, duration, len(audio_data), endpoint=False)
        target_x = np.linspace(0.0, duration, target_length, endpoint=False)
        return np.interp(target_x, source_x, audio_data).astype(np.float32)

    def decode_audio(
        self,
        session_id: str,
        source_id: str,
        file_path: str,
        sample_rate: int = 44100,
        buffer_size: int = 1024,
    ) -> Optional[list]:
        """Decode an uploaded file into mono float32 buffers."""
        try:
            import miniaudio

            decoded = miniaudio.decode_file(file_path)
            audio = np.asarray(decoded.samples, dtype=np.float32)
            if getattr(decoded, 'num_channels', 1) > 1:
                audio = self._stereo_to_mono(audio.reshape(-1, decoded.num_channels))
            if getattr(decoded, 'sample_rate', sample_rate) != sample_rate:
                audio = self._resample_audio(audio, decoded.sample_rate, sample_rate)
            return self._split_into_buffers(audio, buffer_size)
        except Exception as e:
            logging.error(f"Failed to decode audio {file_path}: {e}")
            return None

    def save_uploaded_file_stream(
        self,
        session_id: str,
        source_id: str,
        file_obj,
        filename: str,
    ) -> Optional[str]:
        """save a werkzeug filestorage object to disk."""
        from werkzeug.utils import secure_filename
        try:
            session_dir = os.path.join(self.upload_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            safe_src = secure_filename(source_id)
            safe_orig = secure_filename(filename)
            safe_filename = f"{safe_src}_{safe_orig}"
            file_path = os.path.join(session_dir, safe_filename)
            file_obj.save(file_path)
            if session_id not in self.temp_files:
                self.temp_files[session_id] = []
            self.temp_files[session_id].append(file_path)
            logging.info(f"Saved uploaded file: {file_path}")
            return file_path
        except Exception as e:
            logging.error(f"Failed to save uploaded file (stream): {e}")
            return None

    def get_audio_info(self, file_path: str) -> Optional[dict]:
        """return metadata for an audio file."""
        try:
            filename = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            if PY3DTI_AVAILABLE:
                from miniaudio import get_file_info
                info = get_file_info(file_path)
                return {
                    "filename":              filename,
                    "duration_seconds":      info.duration,
                    "sample_rate":           info.sample_rate,
                    "original_sample_rate":  info.sample_rate,
                    "file_size_bytes":       file_size,
                    "num_frames":            info.num_frames,
                    "channels":              info.nchannels,
                    "estimated_buffers":     int(np.ceil(info.num_frames / 1024)),
                    "format":                str(info.file_format),
                }
            else:
                return {
                    "filename":             filename,
                    "duration_seconds":     0.0,
                    "sample_rate":          44100,
                    "original_sample_rate": 44100,
                    "file_size_bytes":      file_size,
                    "num_frames":           0,
                    "estimated_buffers":    0,
                    "channels":             1,
                    "format":               "unknown",
                }

        except Exception as e:
            logging.error(f"Failed to get audio info for {file_path}: {e}")
            return None

    def convert_mp3_to_audio_buffers(
        self,
        file_path: str,
        sample_rate: int = 44100,
        buffer_size: int = 1024,
    ) -> Optional[list]:
        """decode audio file and return list of mono float32 buffers."""
        try:
            if PY3DTI_AVAILABLE:
                logging.info(f"Decoding audio: {file_path}")
                decoded = decode_file(
                    filename=file_path,
                    output_format=SampleFormat.FLOAT32,
                    nchannels=1,
                    sample_rate=sample_rate,
                )
                audio = np.asarray(decoded.samples, dtype=np.float32)

            else:
                logging.warning(f"miniaudio unavailable, using scipy fallback: {file_path}")
                try:
                    import scipy.io.wavfile as wavfile
                    sr, audio = wavfile.read(file_path)
                    audio = audio.astype(np.float32)
                    if audio.ndim == 2:
                        audio = audio.mean(axis=1).astype(np.float32)
                    if sr != sample_rate:
                        from scipy.signal import resample_poly
                        from fractions import Fraction
                        # keep the resampling ratio bounded so scipy does not build oversized filters.
                        ratio = Fraction(sample_rate, sr).limit_denominator(512)
                        audio = resample_poly(audio, ratio.numerator, ratio.denominator).astype(np.float32)
                except ImportError:
                    logging.error("Neither miniaudio nor scipy available")
                    return None

            remainder = len(audio) % buffer_size
            if remainder:
                audio = np.pad(audio, (0, buffer_size - remainder))
            buffers = list(audio.reshape(-1, buffer_size))

            logging.info(
                f"Decoded {file_path} → {len(buffers)} buffers "
                f"({len(buffers) * buffer_size / sample_rate:.2f}s)"
            )
            return buffers

        except Exception as e:
            logging.error(f"Failed to convert audio {file_path}: {e}")
            return None

    def cleanup_session_files(self, session_id: str):
        """remove all files uploaded by a session."""
        try:
            if session_id not in self.temp_files:
                return
            for file_path in self.temp_files[session_id]:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logging.info(f"Cleaned up file: {file_path}")
            session_dir = os.path.join(self.upload_dir, session_id)
            if os.path.exists(session_dir):
                try:
                    os.rmdir(session_dir)
                except OSError:
                    pass  # non-empty directory; leave for manual cleanup
            del self.temp_files[session_id]
            logging.info(f"Cleaned up session files: {session_id}")
        except Exception as e:
            logging.error(f"Failed to cleanup session files: {e}")

    def encode_buffer_to_base64(self, buffer: np.ndarray) -> str:
        return base64.b64encode(buffer.astype(np.float32).tobytes()).decode("utf-8")

    def get_session_files(self, session_id: str) -> list:
        try:
            session_dir = os.path.join(self.upload_dir, session_id)
            if not os.path.exists(session_dir):
                return []
            files = []
            for fname in os.listdir(session_dir):
                fpath = os.path.join(session_dir, fname)
                if os.path.isfile(fpath):
                    info = self.get_audio_info(fpath)
                    if info:
                        files.append(info)
            return files
        except Exception as e:
            logging.error(f"Failed to get session files: {e}")
            return []
