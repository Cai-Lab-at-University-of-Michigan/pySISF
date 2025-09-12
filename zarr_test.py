from dataclasses import dataclass, replace

from zarr.abc.codec import ArrayBytesCodec
from zarr.core.buffer import Buffer, NDArrayLike, NDBuffer

from zarr.registry import register_codec

import numpy as np
import h5ffmpeg

@dataclass(frozen=True)
class VideoCodec(ArrayBytesCodec):
    if_fixed_size=False

    def __init__(self) -> None:
        pass

    @classmethod
    def from_dict(cls, data):
        #_, configuration_parsed = parse_named_configuration(
        #    data, "bytes", require_configuration=False
        #)
        #configuration_parsed = configuration_parsed or {}
        #return cls(**configuration_parsed)
        return cls()

    def to_dict(self):
        return {"name":"video_codec"}

    def evolve_from_array_spec(self, array_spec):
        return self

    async def _encode_single(self, chunk_array, chunk_spec):
        assert isinstance(chunk_array, NDBuffer)

        nd_array = chunk_array.as_ndarray_like()
        #print("nd_array dtype:", nd_array.dtype, "shape:", nd_array.shape)

        if nd_array.dtype not in (np.uint8, np.uint16):
            raise ValueError("VideoCodec only supports uint8 and uint16 data types.")
        if len(nd_array.shape) != 3:
            raise ValueError("VideoCodec only supports 3D data.")

        #nd_array = nd_array.ravel().view(dtype="b")
        #out = chunk_spec.prototype.buffer.from_array_like(nd_array)

        compressed = h5ffmpeg.compress_native(nd_array)
        return chunk_spec.prototype.buffer.from_bytes(compressed)


    async def _decode_single(self, chunk_bytes, array_spec):
        decompressed = h5ffmpeg.decompress_native(chunk_bytes.to_bytes())
        return array_spec.prototype.nd_buffer.from_ndarray_like(decompressed)

register_codec("videocodec", VideoCodec)
