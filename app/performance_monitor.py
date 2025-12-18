class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.lock = Lock()

        self.latencies: deque = deque(maxlen=window_size)
        self.processing_times: deque = deque(maxlen=window_size)
        self.message_timestamps: deque = deque(maxlen=window_size)

        self.total_messages = 0
        self.total_errors = 0
        self.start_time = time.time()