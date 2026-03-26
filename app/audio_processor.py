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


def _forward_up_to_quaternion(forward: Dict[str, float], up: Dict[str, float]) -> Tuple[float, float, float, float]:
    """convert forward/up vectors to quaternion for py3dti."""
    f = np.array([forward['x'], forward['y'], forward['z']], dtype=np.float64)
    u = np.array([up['x'],      up['y'],      up['z']],      dtype=np.float64)

    f_norm = np.linalg.norm(f)
    u_norm = np.linalg.norm(u)
    if f_norm < 1e-9 or u_norm < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)  # identity
    f /= f_norm
    u /= u_norm

    r = np.cross(f, u)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    r /= r_norm
    u = np.cross(r, f)  # reorthogonalise up

    m = np.array([
        [ r[0],  u[0], -f[0]],
        [ r[1],  u[1], -f[1]],
        [ r[2],  u[2], -f[2]],
    ], dtype=np.float64)

    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return (float(w), float(x), float(y), float(z))


class AudioProcessor:
    """real-time spatial audio processor using py3dti."""

    def __init__(self, sample_rate: int = 48000, buffer_size: int = 1024, max_sources: int = 16):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.max_sources = max_sources
        self.is_initialised = False
        self.placeholder_mode = False

        self.renderer = None
        self.listener = None
        self._streamer = None
        self.sources = {}
        self.source_buffers = {}

        self.listener_pose = {
            'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'orientation': {
                'forward': {'x': 0.0, 'y': 0.0, 'z': -1.0},
                'up':      {'x': 0.0, 'y': 1.0, 'z':  0.0},
            }
        }
        print(f"created audioprocessor: {sample_rate}hz, {buffer_size} samples, max_sources={max_sources}")

    def initialise(self, hrtf_file: str = None, hrtf_path: str = None) -> bool:
        """initialise with hrtf file, falls back to placeholder mode on failure."""
        if not PY3DTI_AVAILABLE:
            print("py3dti not available - using placeholder mode")
            self.placeholder_mode = True
            self.is_initialised = True
            return True

        try:
            print("initialising py3dti renderer...")
            self.renderer = py3dti.BinauralRenderer(
                rate=self.sample_rate,
                buffer_size=self.buffer_size,
                resampled_angular_resolution=5,
            )
            print("BinauralRenderer created")

            self.listener = self.renderer.add_listener(position=None, orientation=None)
            print("listener added")

            if hrtf_path:
                resolved_path = hrtf_path
            elif hrtf_file:
                resolved_path = os.path.join('hrtf_datasets', hrtf_file)
            else:
                print("no hrtf specified - using placeholder mode")
                self.placeholder_mode = True
                self.is_initialised = True
                return True

            abs_path = os.path.abspath(resolved_path)
            print(f"loading hrtf from: {abs_path}")
            if not os.path.exists(abs_path):
                print(f"hrtf file not found: {abs_path}")
                return False
            resolved_path = abs_path

            self.listener.load_hrtf_from_sofa(resolved_path)
            print(f"hrtf loaded: {resolved_path}")

            self.is_initialised = True
            self.placeholder_mode = False
            print("py3dti initialised successfully")
            return True

        except Exception as e:
            print(f"failed to initialise py3dti: {e}")
            print("falling back to placeholder mode")
            self.renderer = None
            self.listener = None
            self._streamer = None
            self.placeholder_mode = True
            self.is_initialised = True
            return True

    def create_source(self, source_id: str, position: Dict[str, float]) -> bool:
        if not self.is_initialised:
            print(f"Cannot create source: processor not initialised")
            return False

        if source_id in self.sources:
            print(f"Source {source_id} already exists")
            return False

        if len(self.sources) >= self.max_sources:
            print(f"Maximum sources limit reached ({self.max_sources})")
            return False

        if self.placeholder_mode:
            self.sources[source_id] = {'position': position}
            self.source_buffers[source_id] = np.zeros(self.buffer_size, dtype=np.float32)
            print(f"created source (placeholder): {source_id} at {position}")
            return True

        try:
            source = self.renderer.add_source(
                position=(position['z'], -position['x'], position['y'])
            )
            # disable unnecessary effects
            source.far_distance_effect                  = False
            source.reverb_processing                    = False
            source.near_field_effect                    = False
            source.propagation_delay                    = False
            source.anechoic_distance_attenuation_smoothing = False
            self.sources[source_id] = source
            self.source_buffers[source_id] = np.zeros(self.buffer_size, dtype=np.float32)
            self._streamer = None  # invalidated: CCore snapshot must be rebuilt on next render
            print(f"created py3dti source: {source_id} at {position}")
            return True

        except Exception as e:
            print(f"Failed to create source: {e}")
            return False

    def update_source_position(self, source_id: str, position: Dict[str, float]) -> bool:
        if source_id not in self.sources:
            print(f"Source {source_id} not found for position update")
            return False

        if self.placeholder_mode:
            self.sources[source_id]['position'] = position
            print(f"updated position (placeholder): {source_id} to {position}")
            return True

        try:
            self.sources[source_id].position = (position['z'], -position['x'], position['y'])
            print(f"updated py3dti source position: {source_id} to {position}")
            return True

        except Exception as e:
            print(f"Failed to update position: {e}")
            return False

    def remove_source(self, source_id: str) -> bool:
        if source_id not in self.sources:
            print(f"Source {source_id} not found for removal")
            return False

        try:
            del self.sources[source_id]
            self.source_buffers.pop(source_id, None)
            self._streamer = None  # invalidated: CCore snapshot must be rebuilt on next render
            print(f"removed source: {source_id}")
            return True

        except Exception as e:
            print(f"Failed to remove source: {e}")
            return False

    def get_source_count(self) -> int:
        return len(self.sources)

    def get_source_ids(self) -> list:
        return list(self.sources.keys())

    def set_listener_pose(
        self,
        position: Dict[str, float],
        orientation: Dict[str, Dict[str, float]],
        position_changed: bool = True
    ) -> bool:
        """update listener position and orientation."""
        self.listener_pose['position'] = position.copy()
        self.listener_pose['orientation'] = {
            'forward': orientation['forward'].copy(),
            'up':      orientation['up'].copy(),
        }

        if not self.is_initialised:
            print("Cannot set listener pose: processor not initialised")
            return False

        if self.placeholder_mode:
            print(f"placeholder mode: set listener pose to {position}, orientation {orientation}")
            return True

        try:
            self.listener.position = (position['x'], position['y'], position['z'])

            quat = _forward_up_to_quaternion(orientation['forward'], orientation['up'])
            self.listener.orientation = quat

            if position_changed:
                print(f"listener position updated: {position}")
            else:
                print(f"listener orientation updated (position unchanged)")

            return True

        except Exception as e:
            print(f"Failed to set listener pose: {e}")
            return False

    def process_sources(
        self, buffers_map: Dict[str, np.ndarray]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """process multiple source buffers and return mixed output."""
        if not self.is_initialised or not buffers_map:
            return None

        silence = np.zeros(self.buffer_size, dtype=np.float32)
        for source_id in self.sources:
            if source_id in buffers_map:
                buffer = np.asarray(buffers_map[source_id], dtype=np.float32)
                if len(buffer) < self.buffer_size:
                    buffer = np.pad(buffer, (0, self.buffer_size - len(buffer)))
                elif len(buffer) > self.buffer_size:
                    buffer = buffer[:self.buffer_size]
                self.source_buffers[source_id] = buffer
            else:
                self.source_buffers[source_id] = silence

        return self._mix_all_sources()

    def process_audio(
        self, source_id: str, audio_buffer: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """process single audio buffer with spatial rendering."""
        if source_id not in self.sources or not self.is_initialised:
            print(f"Cannot process: source {source_id} not found or not initialised")
            return None

        audio_buffer = np.asarray(audio_buffer)
        if audio_buffer.ndim == 2:
            if audio_buffer.shape[1] == 1:
                audio_buffer = audio_buffer.flatten()
            else:
                print(f"Cannot process: invalid input shape {audio_buffer.shape}, expected mono")
                return None
        elif audio_buffer.ndim != 1:
            print(f"Cannot process: unsupported input dimensions {audio_buffer.ndim}")
            return None

        if audio_buffer.dtype != np.float32:
            audio_buffer = audio_buffer.astype(np.float32)

        if len(audio_buffer) < self.buffer_size:
            audio_buffer = np.pad(audio_buffer, (0, self.buffer_size - len(audio_buffer)))
        elif len(audio_buffer) > self.buffer_size:
            audio_buffer = audio_buffer[:self.buffer_size]

        self.source_buffers[source_id] = audio_buffer
        return self._mix_all_sources()

    def _mix_all_sources(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self.sources:
            return None

        start_time = time.time()

        if self.placeholder_mode:
            return self._mix_sources_placeholder(start_time)

        try:
            samples_map  = {}
            position_map = {}
            for source_id, source in self.sources.items():
                buf = self.source_buffers.get(source_id)
                if buf is not None:
                    samples_map[source]  = buf
                    position_map[source] = source.position

            if not samples_map:
                return None

            if self._streamer is None:
                self._streamer = self.renderer.render_online()

            binaural = self._streamer(samples_map, position_map)
            left  = np.ascontiguousarray(binaural[:, 0], dtype=np.float32)
            right = np.ascontiguousarray(binaural[:, 1], dtype=np.float32)

            peak = max(np.max(np.abs(left)), np.max(np.abs(right)))
            if peak > 1.0:
                scale = 1.0 / peak
                left  = left  * scale
                right = right * scale

            processing_time = (time.time() - start_time) * 1000
            print(f"rendered {len(self.sources)} py3dti sources in {processing_time:.2f}ms")
            return left, right

        except Exception as e:
            print(f"failed to mix audio: {e}")
            return None

    def _mix_sources_placeholder(self, start_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """simple stereo panning when py3dti unavailable."""
        mixed_left  = np.zeros(self.buffer_size, dtype=np.float32)
        mixed_right = np.zeros(self.buffer_size, dtype=np.float32)

        listener_pos = self.listener_pose['position']

        for source_id, buffer in self.source_buffers.items():
            if buffer is None or len(buffer) != self.buffer_size:
                continue

            source_pos = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            src = self.sources.get(source_id)
            if isinstance(src, dict):
                source_pos = src.get('position', source_pos)

            dx = source_pos['x'] - listener_pos['x']
            dy = source_pos['y'] - listener_pos['y']
            dz = source_pos['z'] - listener_pos['z']
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)
            attenuation = np.clip(1.0 / (1.0 + distance * 0.1), 0.1, 1.0)

            pan = np.clip(dx / 5.0, -1.0, 1.0)
            left_gain  = np.sqrt(max(0.0, 1.0 - pan)) * 0.707
            right_gain = np.sqrt(max(0.0, 1.0 + pan)) * 0.707

            attenuated   = buffer * attenuation
            mixed_left  += attenuated * left_gain
            mixed_right += attenuated * right_gain

        max_val = np.max(np.abs(np.concatenate([mixed_left, mixed_right])))
        if max_val > 0.95:
            scale = 0.95 / max_val
            mixed_left  *= scale
            mixed_right *= scale

        processing_time = (time.time() - start_time) * 1000
        print(f"mixed {len(self.source_buffers)} sources with spatial simulation in {processing_time:.2f}ms")
        return mixed_left, mixed_right

    def cleanup(self):
        print("cleaning up...")
        self.sources.clear()
        self.source_buffers.clear()
        self.is_initialised = False
        self.placeholder_mode = False
        self.renderer = None
        self.listener = None
        self._streamer = None
        print("cleanup complete")