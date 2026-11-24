"""Thread-safe ring buffer for audio samples.

The mic writes in one thread, the pipeline reads in another. When the
reader falls behind, the oldest samples are dropped instead of growing
memory without bound — a voice agent would rather lose stale audio than
fall minutes behind the speaker.
"""

import threading


class RingBuffer:
    def __init__(self, capacity_samples):
        if capacity_samples <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity_samples
        self._buf = [0] * capacity_samples
        self._head = 0  # next sample to read
        self._size = 0
        self.dropped = 0  # samples thrown away on overflow
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def __len__(self):
        with self._lock:
            return self._size

    @property
    def available(self):
        return len(self)

    def write(self, samples):
        """Append samples; drops the oldest ones if the buffer is full."""
        with self._not_empty:
            for sample in samples:
                if self._size == self.capacity:
                    self._head = (self._head + 1) % self.capacity
                    self._size -= 1
                    self.dropped += 1
                pos = (self._head + self._size) % self.capacity
                self._buf[pos] = sample
                self._size += 1
            self._not_empty.notify_all()

    def read(self, num_samples, timeout=None):
        """Read up to num_samples, blocking until at least one is available.

        Returns fewer than requested only when the buffer is drained and
        the timeout expires (or the timeout is 0). Raises TimeoutError
        when nothing arrives within the timeout.
        """
        with self._not_empty:
            if self._size == 0 and not self._not_empty.wait(timeout):
                raise TimeoutError("no audio arrived within timeout")
            count = min(num_samples, self._size)
            out = [
                self._buf[(self._head + i) % self.capacity] for i in range(count)
            ]
            self._head = (self._head + count) % self.capacity
            self._size -= count
            return out

    def read_available(self):
        """Read everything currently buffered, without blocking."""
        with self._lock:
            count = self._size
            out = [
                self._buf[(self._head + i) % self.capacity] for i in range(count)
            ]
            self._head = (self._head + count) % self.capacity
            self._size = 0
            return out

    def clear(self):
        with self._lock:
            self._head = 0
            self._size = 0
