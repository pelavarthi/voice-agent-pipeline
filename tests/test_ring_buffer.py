import threading

import pytest

from voice.ring_buffer import RingBuffer


def test_write_then_read_returns_fifo_order():
    buf = RingBuffer(10)
    buf.write([1, 2, 3])
    assert buf.read(2) == [1, 2]
    assert buf.read(1) == [3]
    assert len(buf) == 0


def test_read_wraps_around_the_end_of_the_buffer():
    buf = RingBuffer(4)
    buf.write([1, 2, 3, 4])
    assert buf.read(3) == [1, 2, 3]
    buf.write([5, 6, 7])
    assert buf.read_available() == [4, 5, 6, 7]


def test_overflow_drops_oldest_samples_and_counts_them():
    buf = RingBuffer(3)
    buf.write([1, 2, 3, 4, 5])
    assert buf.dropped == 2
    assert buf.read_available() == [3, 4, 5]


def test_read_blocks_until_audio_arrives():
    buf = RingBuffer(10)
    got = []

    def writer():
        buf.write([7, 8])

    threading.Timer(0.05, writer).start()
    got.extend(buf.read(2, timeout=1.0))
    assert got == [7, 8]


def test_read_times_out_on_empty_buffer():
    buf = RingBuffer(10)
    with pytest.raises(TimeoutError):
        buf.read(1, timeout=0.05)


def test_clear_empties_the_buffer():
    buf = RingBuffer(10)
    buf.write([1, 2, 3])
    buf.clear()
    assert len(buf) == 0
    assert buf.read_available() == []
