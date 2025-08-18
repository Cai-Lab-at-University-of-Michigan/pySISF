#   ---------------------------------------------------------------------------------
#   Copyright (c) University of Michigan 2020-2025. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""pySISF."""

import struct
import os
import tqdm
import itertools
import concurrent
import concurrent.futures
from collections import defaultdict

import zstd
import numpy as np

from pySISF import sndif_utils # vidlib
import h5ffmpeg

METADATA_NAME = "metadata.bin"
DEBUG = False

# HEADER_LAYOUT = "<HHHHHHHQQQ"
HEADER_LAYOUT = f"<{'H' * 6}{'Q' * 6}"
HEADER_SIZE = struct.calcsize(HEADER_LAYOUT)
SHARD_HEADER_LAYOUT = f"<{'H' * 7}{'Q' * (3 + 6)}"
SHARD_HEADER_SIZE = struct.calcsize(SHARD_HEADER_LAYOUT)
SHARD_LINE_LAYOUT = f"<QL"
SHARD_LINE_SIZE = struct.calcsize(SHARD_LINE_LAYOUT)

CURRENT_VERSION = 1


def iterate_bounded(max_val, step_size):
    i = 0
    while i < max_val:
        yield (i, min(i + step_size, max_val))
        i += step_size


def create_metadata(version, dtype, channel_count, mchunk, res, size):
    return struct.pack(
        HEADER_LAYOUT,
        version,
        dtype,
        channel_count,
        mchunk[0],
        mchunk[1],
        mchunk[2],
        res[0],
        res[1],
        res[2],
        size[0],
        size[1],
        size[2],
    )


def get_dtype_code(i):
    if i == np.uint16:
        return 1
    if i == np.uint8:
        return 2

    raise TypeError("Unknown Data Type")


def create_shard_worker(data, coords, compression, compression_opts=None, buffer_size=None):
    if buffer_size:
        cs = (coords[1] - coords[0], coords[3] - coords[2], coords[5] - coords[4])

        if cs[0] == buffer_size[0] and cs[1] == buffer_size[1] and cs[2] == buffer_size[2]:
            c = data[coords[0] : coords[1], coords[2] : coords[3], coords[4] : coords[5]]
        else:
            c = np.zeros(shape=buffer_size if buffer_size else cs, dtype=np.uint16)
            c[: cs[0], : cs[1], : cs[2]] += data[coords[0] : coords[1], coords[2] : coords[3], coords[4] : coords[5]]
    else:
        c = data[coords[0] : coords[1], coords[2] : coords[3], coords[4] : coords[5]]

    # compress
    match compression:
        case 0:
            chunk_bin = c.tobytes(order="C")
            return chunk_bin
        case 1:
            chunk_bin = c.tobytes(order="C")
            return zstd.ZSTD_compress(chunk_bin, 9, 1)
        case 2:
            return h5ffmpeg.compress_native(c, codec="libx264")
        case 3:
            return h5ffmpeg.compress_native(c, codec="libsvtav1")
        case _:
            raise ValueError(f"Invalid compression parameter {compression}")


def create_shard(
    fname_data: str,
    fname_meta: str,
    data,
    chunk_size,
    compression,
    compression_opts=None,
    thread_count=8,
    chunk_batch=1024,
    crop=None,
    progress=True,
) -> None:
    """
    Function to create a SISF shard.

    parameters:
        fname_data (str): Name of the data file to create.
        fname_meta (str): Name of the metadata file to create.
        data (3D numpy array-like): Raw data for the chunk.
        chunk_size (3-tuple of int): Size of each chunk
        compression (int): compression codec to use
        thread_count (int, default 8): number of threads to use for data packing
        chunk_batch (int, default 1024): number of jobs to allocate per thread for load balancing
        crop (3-tuple of int, default None): if set, encodes a crop factor into the shard
        progress (bool, default True): prints a loading bar using `tqdm`
    """
    dtype = 1

    total_chunks = 1
    for i in range(3):
        total_chunks *= sum(1 for _ in iterate_bounded(data.shape[i], chunk_size[i]))

    def iter_chunks(executor):
        for istart, iend in iterate_bounded(data.shape[0], chunk_size[0]):
            for jstart, jend in iterate_bounded(data.shape[1], chunk_size[1]):
                for kstart, kend in iterate_bounded(data.shape[2], chunk_size[2]):
                    yield executor.submit(
                        create_shard_worker,
                        data,
                        (istart, iend, jstart, jend, kstart, kend),
                        compression,
                        compression_opts=compression_opts,
                        buffer_size=chunk_size if (compression==2 or compression==3) else None,
                    )

    chunk_table = []
    with open(fname_data, "wb") as fdata:
        with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = iter_chunks(executor)

            with tqdm.tqdm(total=total_chunks) as pb:
                block_size = 512
                while chunk := list(itertools.islice(futures, block_size)):
                    for future in chunk:
                        chunk_bin = future.result()
                        chunk_table.append((fdata.tell(), len(chunk_bin)))
                        fdata.write(chunk_bin)
                    pb.update(len(chunk))

    # Fill crop with default if not specified
    if crop is None:
        crop = (
            0,
            data.shape[0],
            0,
            data.shape[1],
            0,
            data.shape[2],
        )

    # Write shard header
    towrite = bytearray(
        struct.pack(
            SHARD_HEADER_LAYOUT,
            CURRENT_VERSION,
            dtype,
            1,
            compression,
            chunk_size[0],
            chunk_size[1],
            chunk_size[2],
            data.shape[0],
            data.shape[1],
            data.shape[2],
            *crop,
        )
    )

    # Write shard table
    for o, s in chunk_table:
        towrite.extend(struct.pack(SHARD_LINE_LAYOUT, o, s))

    with open(fname_meta, "wb") as fmeta:
        fmeta.write(bytes(towrite))


def create_sisf(
    fname: str,
    data,
    mchunk_size,
    chunk_size,
    res,
    enable_status=True,
    downsampling=None,
    compression=1,
    thread_count=8,
    compression_opts=None,
) -> None:
    """
    Function to create a SISF archive.

    Parameters:
        fname (string): Name of the folder to place the SISF archive into, created if does not exist.
        data (numpy array-like): Represents the data to be converted in CXYZ format.
        mchunk_size (3-tuple): size of metachunks to create, e.g. (2000,2000,2000).
        res (3-tuple): resolution of the dataset in nanometers (nm), e.g. (100,100,100).
        enable_status (bool, default True): If true, print out a loading bar for creation using `tqdm`.
        downsampling (int, default None): How many downsample tiers to generate.
        compression (int, default 1->ZSTD): What compression codec to use.
        thread_count (int, default 8): How many threads to use for data packing.
    """
    if fname.endswith("/"):
        fname = fname[:-1]

    for folder_name in [fname, f"{fname}/data", f"{fname}/meta"]:
        try:
            os.mkdir(folder_name)
        except FileExistsError:
            pass  # folder exists

    if len(data.shape) == 3:
        channel_count = 1
        size = data.shape
    elif len(data.shape) == 4:
        channel_count = data.shape[0]
        size = data.shape[1:]
    else:
        raise ValueError(f"Invalid image dimension size {data.shape}!")

    dtype_code = get_dtype_code(data.dtype)
    # TODO handle dtype errors

    #print(channel_count, size)

    # Create Header
    with open(f"{fname}/{METADATA_NAME}", "wb") as f:
        header = create_metadata(CURRENT_VERSION, dtype_code, channel_count, mchunk_size, res, size)
        f.write(header)

    # Calculate totals
    if enable_status:
        total = 1
        total *= len(list(iterate_bounded(size[0], mchunk_size[0])))
        total *= len(list(iterate_bounded(size[1], mchunk_size[1])))
        total *= len(list(iterate_bounded(size[2], mchunk_size[2])))
        status_bar = tqdm.tqdm(total=total)

    # Create data chunks
    for c in range(channel_count):
        for i, (istart, iend) in enumerate(iterate_bounded(size[0], mchunk_size[0])):
            for j, (jstart, jend) in enumerate(iterate_bounded(size[1], mchunk_size[1])):
                for k, (kstart, kend) in enumerate(iterate_bounded(size[2], mchunk_size[2])):
                    osizei = iend - istart
                    osizej = jend - jstart
                    osizek = kend - kstart

                    # Generate file names
                    chunk_name = f"chunk_{i}_{j}_{k}.{c}.1X"
                    chunk_name_data = f"{fname}/data/{chunk_name}.data"
                    chunk_name_meta = f"{fname}/meta/{chunk_name}.meta"

                    # Make buffer of only this metachunk
                    chunk = np.zeros((osizei, osizej, osizek), dtype=np.uint16)

                    if channel_count == 1:
                        chunk[...] = data[istart:iend, jstart:jend, kstart:kend]
                    elif channel_count > 1:
                        chunk[...] = data[c, istart:iend, jstart:jend, kstart:kend]
                    else:
                        raise ValueError(f"Invalid channel count! ({channel_count})")

                    # Save 1X image
                    create_shard(
                        chunk_name_data,
                        chunk_name_meta,
                        chunk,
                        chunk_size,
                        compression,
                        thread_count=thread_count,
                        compression_opts=compression_opts,
                    )

                    # Perform downsampling
                    if downsampling is not None:
                        downsample_pyramid = [chunk]

                        for scalei in range(downsampling):
                            # convert from 0, 1, 2, etc. -> 1X, 2X, 4X, etc.
                            scale = 2**scalei

                            # Skip basecase, already handled above
                            if scale == 1:
                                continue

                            # Generate downsampled file names
                            new_chunk_name_data = chunk_name_data.replace(".1X.", f".{scale}X.")
                            new_chunk_name_meta = chunk_name_meta.replace(".1X.", f".{scale}X.")

                            chunk_down = np.zeros(
                                shape=(
                                    downsample_pyramid[-1].shape[0] // 2,
                                    downsample_pyramid[-1].shape[1] // 2,
                                    downsample_pyramid[-1].shape[2] // 2,
                                ),
                                dtype=np.uint16,
                            )

                            # calculate downsampled image
                            sndif_utils.downsample(downsample_pyramid[-1], chunk_down)
                            downsample_pyramid.append(chunk_down)

                            # Assign offsets, but not relevant here

                            create_shard(
                                new_chunk_name_data,
                                new_chunk_name_meta,
                                downsample_pyramid[-1],
                                chunk_size,
                                compression,
                                thread_count=thread_count,
                            )

                        del downsample_pyramid

                    if enable_status:
                        status_bar.update(1)

                    del chunk

    if enable_status:
        status_bar.close()


class sisf_chunk:
    def parse_metadata(self):
        with open(self.fname_meta, "rb") as f:
            self.header_bin = f.read(SHARD_HEADER_SIZE)
            self.header = struct.unpack(SHARD_HEADER_LAYOUT, self.header_bin)

            self.header_parsed = {
                "version": self.header[0],
                "dtype": self.header[1],
                "channel_count": self.header[2],
                "compression_type": self.header[3],
                "chunk_size": tuple(self.header[4:7]),
                "size": tuple(self.header[7:10]),
                "crop": tuple(self.header[10:16]),
            }

            if self.cache is not None:
                index_table_bin = f.read()

                i = 0
                iterable = iter(index_table_bin)
                while True:
                    chunk = bytes(itertools.islice(iterable, SHARD_LINE_SIZE))

                    if len(chunk) == 0:
                        break
                    elif len(chunk) < SHARD_LINE_SIZE:
                        raise ValueError(f"Invalid read size {len(chunk)} when loading metadata cache")

                    self.cache[i] = struct.unpack(SHARD_LINE_LAYOUT, chunk)

                    i += 1


        self.version = self.header_parsed["version"]
        self.dtype = self.header_parsed["dtype"]
        self.channel_count = self.header_parsed["channel_count"]
        self.chunk_size = self.header_parsed["chunk_size"]
        self.size = self.header_parsed["size"]
        self.compression_type = self.header_parsed["compression_type"]
        self.crop = self.header_parsed["crop"]

        self.crop = [(self.crop[i * 2], self.crop[(i * 2) + 1]) for i in range(3)]

        self.crop_size = tuple(int(j - i) for i, j in self.crop)

        self.countx = (self.size[0] + self.chunk_size[0] - 1) // self.chunk_size[0]
        self.county = (self.size[1] + self.chunk_size[1] - 1) // self.chunk_size[1]
        self.countz = (self.size[2] + self.chunk_size[2] - 1) // self.chunk_size[2]

        self.chunk_counts = [self.countx, self.county, self.countz]

    def __init__(self, fname_data, fname_meta, parent=None, cache_metadata=False):
        self.parent = parent
        self.fname_data = fname_data
        self.fname_meta = fname_meta
        self.cache = {} if cache_metadata else None

        self.parse_metadata()

    def find_index(self, x, y, z):
        ix = x // self.chunk_size[0]
        iy = y // self.chunk_size[1]
        iz = z // self.chunk_size[2]

        return (ix * self.countz * self.county) + (iy * self.countz) + iz

    def get_metadata(self, idx):
        if self.cache is not None:
            if idx in self.cache:
                return self.cache[idx]

        with open(self.fname_meta, "rb") as f:
            f.seek(SHARD_HEADER_SIZE + (SHARD_LINE_SIZE * idx))
            meta_bin = f.read(SHARD_LINE_SIZE)
            if len(meta_bin) != SHARD_LINE_SIZE:
                raise ValueError(f"Invalid read size {len(meta_bin)}, likely invalid chunk id {idx}")
            read_offset, read_size = struct.unpack(SHARD_LINE_LAYOUT, meta_bin)

        return (read_offset, read_size)

    def get_chunk(self, idx):
        meta_off, meta_size = self.get_metadata(idx)
        with open(self.fname_data, "rb") as f:
            f.seek(meta_off)
            chunk_compressed = f.read(meta_size)

        sx, sy, sz = self.get_chunk_size(idx)

        match self.compression_type:
            case 0:
                chunk_decompressed = chunk_compressed
                out = np.frombuffer(chunk_decompressed, dtype=(np.uint16 if self.dtype == 1 else np.uint8))
                out = out.reshape((sx, sy, sz))
            case 1:
                chunk_decompressed = zstd.decompress(chunk_compressed)
                out = np.frombuffer(chunk_decompressed, dtype=(np.uint16 if self.dtype == 1 else np.uint8))
                out = out.reshape((sx, sy, sz))
            case 2 | 3:
                out = h5ffmpeg.decompress_native(chunk_compressed)
                out = out[:sx, :sy, :sz]  # crop to size, discard padding
            case _:
                raise NotImplementedError(f"Decompression type {self.compression_type} not implemented.")

        return out

    def get_chunk_coords(self, idx):
        dx = idx // (self.countz * self.county)
        dy = (idx - dx * self.countz * self.county) // self.countz
        dz = idx - dx * self.countz * self.county - dy * self.countz

        return (dx, dy, dz)

    def get_chunk_size(self, idx):
        chunk_coords = self.get_chunk_coords(idx)

        return tuple(  # istart, iend, isize
            min((chunk_coords[i] + 1) * self.chunk_size[i], self.size[i]) - chunk_coords[i] * self.chunk_size[i]
            for i in range(3)
        )

    def get_chunk_numpy(self, idx):
        # This function is here for compatibility with legacy code
        return self.get_chunk(idx)

    def read_pixel(self, x, y, z):
        xmin = self.chunk_size[0] * (x // self.chunk_size[0])
        xmax = min(self.size[0], xmin + self.chunk_size[0])
        xsize = xmax - xmin

        ymin = self.chunk_size[1] * (y // self.chunk_size[1])
        ymax = min(self.size[1], ymin + self.chunk_size[1])
        ysize = ymax - ymin

        zmin = self.chunk_size[2] * (z // self.chunk_size[2])
        zmax = min(self.size[2], zmin + self.chunk_size[2])
        zsize = zmax - zmin

        coffset = ((x - xmin) * ysize * zsize) + ((y - ymin) * zsize) + (z - zmin)
        chunk_id = self.find_index(x, y, z)
        chunk = self.get_chunk(chunk_id)

        return chunk[coffset]

    def __getitem__(self, key):
        if len(key) != 3:
            raise AttributeError("Array access must specify all 3 dimensions.")

        for s in key:
            if type(s) is slice:
                if s.step is not None:
                    raise AttributeError("Stepped selection is not supported.")

        keys = []
        for i, a in enumerate(key):
            if type(a) is int:
                keys.append((a, a + 1))
            elif type(a) is slice:
                keys.append(
                    (
                        a.start if a.start is not None else 0,
                        a.stop if a.stop is not None else self.shape[i],
                    )
                )
            else:
                raise NotImplementedError("Unknown selector type")
        key = keys

        for i, (start, stop) in enumerate(key):
            if stop < start:
                raise AttributeError("Incorrect parameter ordering.")
            if start < 0 or stop < 0:
                raise NotImplementedError("Negative indexing not implemented.")

            if stop > self.shape[i] or start >= self.shape[i]:
                raise IndexError(f"Axis {i} selection ({start, stop}) out of range ({self.shape[i]}).")

        # Define output variable
        outshape = tuple(stop - start for start, stop in keys)
        out = np.zeros(shape=outshape, dtype=np.uint16)  # TODO should dynamically change dtype

        # Shift stop and start to match crop
        keys = tuple((start + crop_start, stop + crop_start) for (crop_start, _), (start, stop) in zip(self.crop, key))

        xstart = 0
        for (cxstart, _), (sxstart, sxend) in sisf_chunk.iterate_chunks(key[0][0], key[0][1], self.chunk_size[0]):
            xsize = sxend - sxstart
            ystart = 0
            for (cystart, _), (systart, syend) in sisf_chunk.iterate_chunks(key[1][0], key[1][1], self.chunk_size[1]):
                ysize = syend - systart
                zstart = 0
                for (czstart, _), (szstart, szend) in sisf_chunk.iterate_chunks(
                    key[2][0], key[2][1], self.chunk_size[2]
                ):
                    zsize = szend - szstart

                    chunk_id = self.find_index(cxstart, cystart, czstart)
                    chunk = self.get_chunk_numpy(chunk_id)

                    out[
                        xstart : xstart + xsize,
                        ystart : ystart + ysize,
                        zstart : zstart + zsize,
                    ] = chunk[sxstart:sxend, systart:syend, szstart:szend]

                    del chunk

                    zstart += zsize
                ystart += ysize
            xstart += xsize

        return out

    def __repr__(self):
        return f"<sif chunk {self.fname_data}/{self.fname_meta} {self.shape}>"

    @staticmethod
    def iterate_chunks(rstart, rstop, cs):
        for cstart in range(cs * (rstart // cs), cs * ((rstop + cs - 1) // cs), cs):
            cend = cstart + cs

            sstart = max(cstart, rstart) - cstart
            send = min(cend, rstop) - cstart

            yield ((cstart, cend), (sstart, send))

    @property
    def shape(self):
        return self.crop_size


class sisf:
    def parse_metadata(self):
        with open(f"{self.fname}/{METADATA_NAME}", "rb") as f:
            self.header_bin = f.read(HEADER_SIZE)
            self.header = struct.unpack(HEADER_LAYOUT, self.header_bin)

            self.header_parsed = {
                "version": self.header[0],
                "dtype": self.header[1],
                "channel_count": self.header[2],
                "mchunk": self.header[3:6],
                "res": self.header[6:9],
                "size": self.header[9:12],
            }

        self.version = self.header_parsed["version"]
        self.dtype = self.header_parsed["dtype"]
        self.channel_count = self.header_parsed["channel_count"]
        self.mchunk = self.header_parsed["mchunk"]
        self.res = self.header_parsed["res"]
        self.size = self.header_parsed["size"]

    def __init__(self, fname, cache_metadata=False):
        self.fname = fname
        if self.fname.endswith("/"):
            self.fname = self.fname[:-1]

        self.cache_metadata = cache_metadata

        # self.chunk_lock = thread.Mutex()

        self.parse_metadata()

    @property
    def shape(self):
        return (self.channel_count, *self.size)

    def get_chunk(self, x, y, z, c, s):
        chunk_fname = f"chunk_{x}_{y}_{z}.{c}.{s}X"
        fname_data = f"{self.fname}/data/{chunk_fname}.data"
        fname_meta = f"{self.fname}/meta/{chunk_fname}.meta"

        return sisf_chunk(fname_data, fname_meta, parent=self, cache_metadata=self.cache_metadata)

    def __getitem__(self, key):
        if len(key) != 4:
            raise AttributeError("Array access must specify all 4 dimensions.")

        for s in key:
            if type(s) is slice:
                if s.step is not None:
                    raise AttributeError("Stepped selection is not supported.")

        keys = []
        for i, a in enumerate(key):
            if type(a) is int:
                keys.append((a, a + 1))
            elif type(a) is slice:
                keys.append(
                    (
                        a.start if a.start is not None else 0,
                        a.stop if a.stop is not None else self.shape[i],
                    )
                )
            else:
                raise NotImplementedError("Unknown selector type")
        key = keys

        for i, (start, stop) in enumerate(key):
            if stop < start:
                raise AttributeError("Incorrect parameter ordering.")
            if start < 0 or stop < 0:
                raise NotImplementedError("Negative indexing not implemented.")

            if i == 0:  # channel
                if stop > self.channel_count or start >= self.channel_count:
                    raise IndexError(f"Channel out of range {(start,stop)}.")
            else:
                if stop > self.shape[i] or start >= self.shape[i]:
                    raise IndexError(f"Axis {i} selection ({start, stop}) out of range ({self.shape[i]}).")

        outshape = tuple(stop - start for start, stop in keys)

        out = np.zeros(shape=outshape, dtype=np.uint16)

        scale = 1
        mcx = self.mchunk[0] // scale
        mcy = self.mchunk[1] // scale
        mcz = self.mchunk[2] // scale

        for c in range(*key[0]):
            xstart = 0
            for (cxstart, _), (sxstart, sxend) in sisf_chunk.iterate_chunks(key[1][0], key[1][1], mcx):
                xsize = sxend - sxstart
                chunk_id_x = cxstart // mcx
                ystart = 0
                for (cystart, _), (systart, syend) in sisf_chunk.iterate_chunks(key[2][0], key[2][1], mcy):
                    ysize = syend - systart
                    chunk_id_y = cystart // mcy
                    zstart = 0
                    for (czstart, _), (szstart, szend) in sisf_chunk.iterate_chunks(key[3][0], key[3][1], mcz):
                        zsize = szend - szstart
                        chunk_id_z = czstart // mcz

                        chunk = self.get_chunk(chunk_id_x, chunk_id_y, chunk_id_z, c, scale)

                        out[
                            c,
                            xstart : xstart + xsize,
                            ystart : ystart + ysize,
                            zstart : zstart + zsize,
                        ] = chunk[sxstart:sxend, systart:syend, szstart:szend]

                        zstart += zsize
                    ystart += ysize
                xstart += xsize

        return out

    def __setitem__(self, key, value):
        raise NotImplementedError("SISF files can not be modified.")

    def __repr__(self):
        size = [f"{i}/{j}/{k}nm" for i, j, k in zip(self.size, self.mchunk, self.res)]
        return f"<sisf archive at {self.fname} ({' x '.join(size)})>"
