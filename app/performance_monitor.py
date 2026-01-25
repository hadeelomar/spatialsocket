import time
from collections import deque
from threading import Lock
from typing import Dict, Any, Optional

class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.lock = Lock()

        self.latencies: deque = deque(maxlen=window_size)
        self.processing_times: deque = deque(maxlen=window_size)
        self.message_timestamps: deque = deque(maxlen=window_size)

        self.total_messages = 0
        self.total_errors = 0
        self.frames_processed = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self.start_time = time.time()
        
        # Per-session metrics
        self.session_metrics: Dict[str, Dict[str, Any]] = {}

    def record_receive_timestamp(self, session_id: str):
        """Record when a message is received"""
        with self.lock:
            timestamp = time.time()
            self.message_timestamps.append(timestamp)
            self.total_messages += 1
            
            # Initialise session metrics if not exists
            if session_id not in self.session_metrics:
                self.session_metrics[session_id] = {
                    'messages': 0,
                    'frames_processed': 0,
                    'bytes_in': 0,
                    'bytes_out': 0,
                    'last_activity': timestamp
                }
            
            self.session_metrics[session_id]['messages'] += 1
            self.session_metrics[session_id]['last_activity'] = timestamp

    def record_processing_start(self, session_id: str):
        """Record when processing starts"""
        with self.lock:
            timestamp = time.time()
            if session_id not in self.session_metrics:
                self.session_metrics[session_id] = {
                    'messages': 0,
                    'frames_processed': 0,
                    'bytes_in': 0,
                    'bytes_out': 0,
                    'last_activity': timestamp,
                    'processing_start': timestamp
                }
            else:
                self.session_metrics[session_id]['processing_start'] = timestamp

    def record_processing_end(self, session_id: str):
        """Record when processing ends and calculate processing time"""
        with self.lock:
            end_time = time.time()
            
            if session_id in self.session_metrics and 'processing_start' in self.session_metrics[session_id]:
                start_time = self.session_metrics[session_id]['processing_start']
                processing_time = end_time - start_time
                self.processing_times.append(processing_time)
                
                # Clean up the start timestamp
                del self.session_metrics[session_id]['processing_start']

    def record_send_timestamp(self, session_id: str):
        """Record when a response is sent"""
        with self.lock:
            timestamp = time.time()
            
            # Calculate latency if we have a receive timestamp
            if self.message_timestamps:
                receive_time = self.message_timestamps[-1]
                latency = timestamp - receive_time
                self.latencies.append(latency)

    def increment_frames_processed(self, session_id: str, count: int = 1):
        """Increment frame counter"""
        with self.lock:
            self.frames_processed += count
            if session_id in self.session_metrics:
                self.session_metrics[session_id]['frames_processed'] += count

    def increment_bytes_in(self, session_id: str, count: int):
        """Increment incoming bytes counter"""
        with self.lock:
            self.bytes_in += count
            if session_id in self.session_metrics:
                self.session_metrics[session_id]['bytes_in'] += count

    def increment_bytes_out(self, session_id: str, count: int):
        """Increment outgoing bytes counter"""
        with self.lock:
            self.bytes_out += count
            if session_id in self.session_metrics:
                self.session_metrics[session_id]['bytes_out'] += count

    def increment_errors(self):
        """Increment error counter"""
        with self.lock:
            self.total_errors += 1

    def get_average_latency(self) -> float:
        """Get average latency over the window"""
        with self.lock:
            if not self.latencies:
                return 0.0
            return sum(self.latencies) / len(self.latencies)

    def get_average_processing_time(self) -> float:
        """Get average processing time over the window"""
        with self.lock:
            if not self.processing_times:
                return 0.0
            return sum(self.processing_times) / len(self.processing_times)

    def get_messages_per_second(self) -> float:
        """Get messages per second over the window"""
        with self.lock:
            if not self.message_timestamps or len(self.message_timestamps) < 2:
                return 0.0
            
            time_span = self.message_timestamps[-1] - self.message_timestamps[0]
            if time_span <= 0:
                return 0.0
            
            return len(self.message_timestamps) / time_span

    def to_dict(self) -> Dict[str, Any]:
        """Export all metrics as a dictionary"""
        with self.lock:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Calculate averages
            avg_latency = self.get_average_latency()
            avg_processing_time = self.get_average_processing_time()
            messages_per_second = self.get_messages_per_second()
            
            return {
                'global': {
                    'uptime_seconds': uptime,
                    'total_messages': self.total_messages,
                    'total_errors': self.total_errors,
                    'frames_processed': self.frames_processed,
                    'bytes_in': self.bytes_in,
                    'bytes_out': self.bytes_out,
                    'average_latency': avg_latency,
                    'average_processing_time': avg_processing_time,
                    'messages_per_second': messages_per_second,
                    'window_size': self.window_size,
                    'current_timestamp': current_time
                },
                'sessions': dict(self.session_metrics),
                'window_stats': {
                    'latency_samples': len(self.latencies),
                    'processing_time_samples': len(self.processing_times),
                    'message_timestamp_samples': len(self.message_timestamps)
                }
            }

    def cleanup_session(self, session_id: str):
        """Remove metrics for a disconnected session"""
        with self.lock:
            if session_id in self.session_metrics:
                del self.session_metrics[session_id]