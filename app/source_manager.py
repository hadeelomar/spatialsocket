"""
Source manager that handles uploaded audio files and converts them for processing
"""
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
    """Manages uploaded audio files and converts them for processing"""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        self.temp_files = {}
        
        os.makedirs(upload_dir, exist_ok=True)
        
        logging.info(f"SourceManager initialised with upload_dir: {upload_dir}")
    
    def save_uploaded_file(self, session_id: str, source_id: str, file_data: bytes, filename: str) -> Optional[str]:
        """
        Save uploaded file and return the path
        
        Args:
            session_id: Session identifier
            source_id: Source identifier  
            file_data: Raw file data
            filename: Original filename
            
        Returns:
            str: Path to saved file or None if failed
        """
        try:
            # Create session-specific directory
            session_dir = os.path.join(self.upload_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)
            
            # Generate safe filename
            safe_filename = f"{source_id}_{filename}"
            file_path = os.path.join(session_dir, safe_filename)
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            # Track for cleanup
            if session_id not in self.temp_files:
                self.temp_files[session_id] = []
            self.temp_files[session_id].append(file_path)
            
            logging.info(f"Saved uploaded file: {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Failed to save uploaded file: {e}")
            return None
    
    def convert_mp3_to_audio_buffers(self, file_path: str, sample_rate: int = 48000, buffer_size: int = 1024) -> Optional[list]:
        """
        Convert audio file to list of audio buffers using py3dti/miniaudio
        
        Args:
            file_path: Path to audio file
            sample_rate: Target sample rate
            buffer_size: Size of each buffer
            
        Returns:
            list: List of audio buffers (numpy arrays) or None if failed
        """
        try:
            if PY3DTI_AVAILABLE:
                logging.info(f"Loading audio with py3dti/miniaudio: {file_path}")
                decoded_file = decode_file(filename=file_path, output_format=SampleFormat.FLOAT32)
                audio = np.asarray(decoded_file.samples)
                
                # Resample if necessary
                if decoded_file.sample_rate != sample_rate:
                    logging.info(f"Resampling from {decoded_file.sample_rate}Hz to {sample_rate}Hz")
                    # Simple resampling - for production, use better resampling
                    resample_ratio = sample_rate / decoded_file.sample_rate
                    audio = np.interp(
                        np.linspace(0, len(audio) - 1, int(len(audio) * resample_ratio)),
                        np.arange(len(audio)),
                        audio
                    )[:int(len(audio) * resample_ratio)]
            else:
                # Fallback: basic audio loading using scipy
                logging.warning(f"py3dti not available, using fallback audio loading: {file_path}")
                try:
                    import scipy.io.wavfile as wavfile
                    audio, sr = wavfile.read(file_path, dtype=np.float32)
                    if sr != sample_rate:
                        # Simple resampling
                        resample_ratio = sample_rate / sr
                        audio = np.interp(
                            np.linspace(0, len(audio) - 1, int(len(audio) * resample_ratio)),
                            np.arange(len(audio)),
                            audio
                        )[:int(len(audio) * resample_ratio)]
                except ImportError:
                    logging.error("Neither py3dti nor scipy available for audio loading")
                    return None
            
            # Convert to float32 if needed
            if audio.dtype != np.float32:
                original_dtype = audio.dtype
                audio = audio.astype(np.float32)
                logging.info(f"Converted audio from {original_dtype} to float32")
            
            # Split into buffers
            buffers = []
            for i in range(0, len(audio), buffer_size):
                buffer = audio[i:i + buffer_size]
                
                # Pad if necessary
                if len(buffer) < buffer_size:
                    buffer = np.pad(buffer, (0, buffer_size - len(buffer)))
                
                buffers.append(buffer)
            
            logging.info(f"Converted {file_path} to {len(buffers)} buffers using py3dti/miniaudio")
            return buffers
            
        except Exception as e:
            logging.error(f"Failed to convert audio {file_path}: {e}")
            return None
    
    def get_audio_info(self, file_path: str) -> Optional[dict]:
        """
        Get information about an audio file using py3dti/miniaudio
        
        Args:
            file_path: Path to audio file
            
        Returns:
            dict: Audio information or None if failed
        """
        try:
            if PY3DTI_AVAILABLE:
                logging.info(f"Getting audio info with py3dti/miniaudio: {file_path}")
                decoded_file = decode_file(filename=file_path)
                
                return {
                    'filename': os.path.basename(file_path),
                    'duration_seconds': decoded_file.duration,
                    'sample_rate': decoded_file.sample_rate,
                    'original_sample_rate': decoded_file.sample_rate,
                    'file_size_bytes': os.path.getsize(file_path),
                    'num_samples': len(decoded_file.samples),
                    'estimated_buffers': int(np.ceil(len(decoded_file.samples) / 1024)),
                    'channels': decoded_file.channels,
                    'format': decoded_file.format
                }
            else:
                # Fallback: basic file info
                logging.warning(f"py3dti not available, using basic file info: {file_path}")
                return {
                    'filename': os.path.basename(file_path),
                    'duration_seconds': 0.0,
                    'sample_rate': 48000,
                    'original_sample_rate': 48000,
                    'file_size_bytes': os.path.getsize(file_path),
                    'num_samples': 0,
                    'estimated_buffers': 0,
                    'channels': 1,
                    'format': 'unknown'
                }
                
        except Exception as e:
            logging.error(f"Failed to get audio info for {file_path}: {e}")
            return None
    
    def cleanup_session_files(self, session_id: str):
        """Clean up all files for a session"""
        try:
            if session_id in self.temp_files:
                for file_path in self.temp_files[session_id]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logging.info(f"Cleaned up file: {file_path}")
                
                # Remove session directory
                session_dir = os.path.join(self.upload_dir, session_id)
                if os.path.exists(session_dir):
                    os.rmdir(session_dir)
                
                del self.temp_files[session_id]
                logging.info(f"Cleaned up all files for session: {session_id}")
                
        except Exception as e:
            logging.error(f"Failed to cleanup session files: {e}")
    
    def encode_buffer_to_base64(self, buffer: np.ndarray) -> str:
        """Encode audio buffer to base64"""
        audio_bytes = buffer.astype(np.float32).tobytes()
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def get_session_files(self, session_id: str) -> list:
        """Get list of files for a session"""
        try:
            session_dir = os.path.join(self.upload_dir, session_id)
            if not os.path.exists(session_dir):
                return []
            
            files = []
            for filename in os.listdir(session_dir):
                file_path = os.path.join(session_dir, filename)
                if os.path.isfile(file_path):
                    info = self.get_audio_info(file_path)
                    if info:
                        files.append(info)
            
            return files
            
        except Exception as e:
            logging.error(f"Failed to get session files: {e}")
            return []
