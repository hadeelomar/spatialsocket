"""source manager for audio files and conversion."""
import os
import numpy as np
import base64
import logging
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