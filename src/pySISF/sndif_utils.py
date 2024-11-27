#   ---------------------------------------------------------------------------------
#   Copyright (c) University of Michigan 2020-2025. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------

import zipfile
import zstd
import tqdm
import concurrent

import numpy as np
from numba import njit


def load_from_zip(
    file_name,
    stack_size=2000,
    stack_select=None,
    thread_count=1,
):
    """
    Load image frames from a SNDIF ZIP archive.

    Parameters:
        file_name (str): Name of the input ZIP file.
        stack_size (int): Number of frames expected.
        stack_select (slice): A parameter passed to the file list to select inputs.
        thread_count (int): Number of threads to launch on the ThreadPoolExecutor.

    Returns:
        Numpy array containing the loaded frames
    """
    zf = zipfile.ZipFile(file_name, mode="r")

    file_list = list(zf.namelist())
    file_list.sort(key=lambda x: int(x.split("_")[1]))

    if stack_select is not None:
        file_list = file_list[stack_select]

    out = bytearray()
    n = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        for r in tqdm.tqdm(
            executor.map(lambda x: zstd.ZSTD_uncompress(zf.read(x)), file_list),
            total=len(file_list),
        ):
            out += r
            n += 1
    zf.close()

    while n < stack_size:
        n += 1
        out += b"0" * (2 * 2304 * 2304)

    outnp = np.frombuffer(out, dtype=np.uint16)
    outnp = outnp.reshape(stack_size, 2304, 2304)

    outnp = np.moveaxis(outnp, 0, -1)

    return outnp


@njit
def downsample(in_array, out_array):
    """
    Utility function to downsample a 3D image by a factor of 2X in all dimensions.

    Parameters:
        in_array (numpy): 3D array containing the input image
        out_array (numpy): identical-sized 3D array to write the output to

    Returns:
        Numpy array containing the downsampled image
    """
    for i, j in zip(in_array.shape, out_array.shape):
        if i // 2 != j:
            raise ValueError(f"Invalid casting ({i}/2) != ({j})")

    for i in range(out_array.shape[0]):
        for j in range(out_array.shape[1]):
            for k in range(out_array.shape[2]):
                total: float = 0.0
                n: int = 0

                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            total += in_array[(i * 2) + ii, (j * 2) + jj, (k * 2) + kk]
                            n += 1

                out_array[i, j, k] = int(total / n)
