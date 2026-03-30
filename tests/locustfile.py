"""
spatialsocket load test - simulates a full user session:
  connect → init audio → create source → upload file → stream → move source → disconnect

run with:
  locust -f tests/locustfile.py --host http://localhost:5000

then open http://localhost:8089, set users=100, ramp=10/s.
"""

import io
import struct
import math
import time
import random
import logging
import requests
import gevent
import gevent.event

import socketio as sio_lib
from locust import User, task, between

logger = logging.getLogger(__name__)


# synthetic 440 hz wav (2s, mono, 48khz, 16-bit)

def _make_wav(frequency=440, duration=2.0, sample_rate=48000) -> bytes:
    num_samples = int(sample_rate * duration)
    buf = io.BytesIO()
    data_size = num_samples * 2
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + data_size))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<IHHIIHH', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', data_size))
    for i in range(num_samples):
        sample = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
        buf.write(struct.pack('<h', sample))
    return buf.getvalue()


_WAV_BYTES = _make_wav()


def _fire(env, name, start, exception=None, length=0):
    """helper to report a timing to locust."""
    env.events.request.fire(
        request_type='socketio',
        name=name,
        response_time=(time.time() - start) * 1000,
        response_length=length,
        exception=exception,
    )


class SpatialSocketUser(User):
    """simulates one concurrent user of the spatialsocket ws api."""

    wait_time = between(1, 3)

    def on_start(self):
        self._sio = sio_lib.Client(logger=False, engineio_logger=False, request_timeout=30)
        self._source_id = f"src-{random.randint(10000, 99999)}"
        self._filename = None
        self._chunk_count = 0

        # events used to synchronise async server responses
        self._ev_init     = gevent.event.Event()
        self._ev_create   = gevent.event.Event()
        self._ev_pos      = gevent.event.Event()
        self._ev_chunk    = gevent.event.Event()

        self._init_ok     = False
        self._create_ok   = False

        self._register_handlers()
        self._connect()

        if self._sio.connected:
            self._run_session()

    def on_stop(self):
        try:
            if self._sio.connected:
                self._sio.disconnect()
        except Exception:
            pass

    # connection

    def _connect(self):
        start = time.time()
        try:
            self._sio.connect(
                self.host,
                transports=['websocket'],
                wait_timeout=10,
            )
            _fire(self.environment, 'connect', start)
        except Exception as e:
            _fire(self.environment, 'connect', start, exception=e)

    # socket.io event handlers

    def _register_handlers(self):
        @self._sio.on('status')
        def on_status(data):
            code = data.get('code', '')

            if code in ('AUDIO_INITIALISED', 'PLACEHOLDER_MODE'):
                self._init_ok = True
                self._ev_init.set()
            elif code == 'INIT_FAILED':
                self._ev_init.set()

            elif code == 'SOURCE_CREATED':
                self._create_ok = True
                self._ev_create.set()
            elif code == 'CREATE_FAILED':
                self._ev_create.set()

            elif code in ('POSITION_UPDATED', 'UPDATE_FAILED'):
                self._ev_pos.set()

        @self._sio.on('file_audio_chunk')
        def on_chunk(data):
            self._chunk_count += 1
            if not self._ev_chunk.is_set():
                self._ev_chunk.set()

        @self._sio.on('error')
        def on_error(data):
            logger.debug(f"server error: {data}")
            # unblock any pending waits so the session doesn't hang
            for ev in (self._ev_init, self._ev_create, self._ev_pos, self._ev_chunk):
                ev.set()

    # session steps

    def _run_session(self):
        try:
            self._step_init_audio()
            if not self._init_ok:
                return

            self._step_create_source()
            if not self._create_ok:
                return

            self._step_upload()
            if not self._filename:
                return

            self._step_stream()
            self._step_move_source()

        except Exception as e:
            logger.error(f"session error: {e}")

    def _step_init_audio(self):
        self._ev_init.clear()
        start = time.time()
        self._sio.emit('init_audio', {'hrtf_file': 'p0200.sofa'})
        ok = self._ev_init.wait(timeout=60)
        _fire(self.environment, 'init_audio', start,
              exception=None if ok and self._init_ok else TimeoutError('init_audio timed out'))

    def _step_create_source(self):
        self._ev_create.clear()
        start = time.time()
        self._sio.emit('create_source', {
            'source_id': self._source_id,
            'position': {'x': random.uniform(-3, 3), 'y': 0, 'z': random.uniform(-3, 3)},
        })
        ok = self._ev_create.wait(timeout=10)
        _fire(self.environment, 'create_source', start,
              exception=None if ok and self._create_ok else TimeoutError('create_source timed out'))

    def _step_upload(self):
        sid = self._sio.sid
        if not sid:
            return
        start = time.time()
        try:
            resp = requests.post(
                f"{self.host}/upload/{sid}/{self._source_id}",
                files={'file': ('tone.wav', _WAV_BYTES, 'audio/wav')},
                timeout=15,
            )
            elapsed = (time.time() - start) * 1000
            if resp.status_code == 200:
                self._filename = resp.json().get('filename')
                self.environment.events.request.fire(
                    request_type='http',
                    name='upload_audio',
                    response_time=elapsed,
                    response_length=len(resp.content),
                    exception=None,
                )
            else:
                self.environment.events.request.fire(
                    request_type='http',
                    name='upload_audio',
                    response_time=elapsed,
                    response_length=0,
                    exception=Exception(f"upload {resp.status_code}"),
                )
        except Exception as e:
            _fire(self.environment, 'upload_audio', start, exception=e)

    def _step_stream(self):
        self._ev_chunk.clear()
        self._chunk_count = 0
        start = time.time()

        self._sio.emit('stream_uploaded_file', {
            'source_id': self._source_id,
            'filename': self._filename,
        })

        ok = self._ev_chunk.wait(timeout=10)
        _fire(self.environment, 'stream→first_chunk', start,
              exception=None if ok else TimeoutError('no first chunk'))

        # let it stream for 3 seconds then stop
        gevent.sleep(3)
        self._sio.emit('stop_file_stream', {})

    def _step_move_source(self):
        for i in range(10):
            self._ev_pos.clear()
            start = time.time()
            self._sio.emit('update_position', {
                'source_id': self._source_id,
                'position': {
                    'x': math.sin(i * 0.6) * 3,
                    'y': 0,
                    'z': math.cos(i * 0.6) * 3,
                },
                '_seq': i,
            })
            ok = self._ev_pos.wait(timeout=5)
            _fire(self.environment, 'update_position', start,
                  exception=None if ok else TimeoutError('update_position timed out'))
            gevent.sleep(0.05)

    @task
    def keep_alive(self):
        # session already ran in on_start; this just keeps the user alive in locust
        gevent.sleep(random.uniform(1, 3))
