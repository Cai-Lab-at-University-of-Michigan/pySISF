"""Compression code 4: zstd level 3 after a byte shuffle."""

from __future__ import annotations

import numpy as np

from pySISF import sisf


def test_byte_shuffle_round_trip() -> None:
    rng = np.random.default_rng(0)
    for n in (0, 1, 2, 7, 64, 1001):
        buf = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        for itemsize in (1, 2, 4):
            shuffled = sisf.byte_shuffle(buf, itemsize)
            assert len(shuffled) == len(buf)
            assert sisf.byte_unshuffle(shuffled, itemsize) == buf


def test_byte_shuffle_groups_low_then_high_bytes() -> None:
    vals = np.array([0x0102, 0x0304, 0x0506], dtype="<u2")
    assert sisf.byte_shuffle(vals.tobytes(), 2) == bytes([0x02, 0x04, 0x06, 0x01, 0x03, 0x05])


def test_code_4_shard_reads_back_bit_exact(tmp_path) -> None:
    rng = np.random.default_rng(4)
    vol = (100 + rng.poisson(9, (70, 50, 33))).astype(np.uint16)
    for code in (1, 4):
        data, meta = str(tmp_path / f"s{code}.data"), str(tmp_path / f"s{code}.meta")
        sisf.create_shard(data, meta, vol, (32, 32, 16), code, progress=False, crop=(0, 70, 0, 50, 0, 33))
        c = sisf.sisf_chunk(data, meta)
        assert c.compression_type == code
        for i in range(int(np.prod(c.chunk_counts))):
            sl = tuple(slice(o * s, min(o * s + s, n)) for o, s, n in zip(c.get_chunk_coords(i), c.chunk_size, c.size))
            assert np.array_equal(c.get_chunk(i), vol[sl])
