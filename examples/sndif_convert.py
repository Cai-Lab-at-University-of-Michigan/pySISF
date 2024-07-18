import sys

sys.version

import tqdm
import pickle
import tifffile
import time

import numpy as np
from scipy import ndimage

import pySISF.sisf
import pySISF.sndif_utils

from basicpy import BaSiC

with open("/home/loganaw/dev/pySISF/070124_flatfield_fits.p", "rb") as f:
    flatfield_fits = pickle.loads(f.read())

files = sys.argv[1:]
if len(files) == 0:
    raise AssertionError("At least one file must be input.")

pySISF.sisf.DEBUG = False

out_files = list((x + ".data", x + ".meta") for x in (x.replace(".zip", "") for x in files))

start = time.time()

for i, (in_file, (out_file_data, out_file_meta)) in enumerate(zip(files, out_files)):
    dt = time.time() - start
    print(f'[{dt:.2f}, {dt / (i+1):.2f}/file] ({i+1}/{len(files)} = {100*(i+1)/len(files):.2f}%) "{in_file}"')

    channel = int(in_file.split("ch")[1].split(".")[0])

    r = pySISF.sndif_utils.load_from_zip(
        in_file,
        stack_size=2000,  # stack_size=10,
        # stack_select=slice(1000, 1010),
        thread_count=16,
        chunk_batch=1,
    )

    if False:
        flatfield = flatfield_fits[channel]
        for i in tqdm.trange(r.shape[2]):
            r[..., i] = flatfield_fits[channel].transform(r[..., i])

    if False:
        import tifffile

        tifffile.imwrite("ff.tif", r.flatfield)
        tifffile.imwrite("df.tif", r.darkfield)

    # flip image for 642 channel :(
    if channel == 1:
        r = r[::-1, ...]

    downsample_pyramid = [r]

    for downsample_rate in range(5):  # 1X 2X 4X 8X 16X
        scale = 2**downsample_rate

        if scale == 1:
            offset = (2304 - 2000) // 2
            crop = (offset, 2304 - offset, offset, 2304 - offset, 0, 2000)
            pySISF.sisf.create_shard(
                out_file_data.replace(".data", ".1X.data"),
                out_file_meta.replace(".meta", ".1X.meta"),
                r,
                (32, 32, 10),
                1,
                crop=crop,
                thread_count=8,
            )
        else:
            downsample_shape = tuple(i // scale for i in r.shape)
            downsample_image = np.zeros(shape=downsample_shape, dtype=np.uint16)

            offset = int((2304 - 2000) / 2 / scale)
            crop = (
                offset,
                downsample_shape[0] - offset,
                offset,
                downsample_shape[1] - offset,
                0,
                downsample_shape[2],
            )

            pySISF.sndif_utils.downsample(downsample_pyramid[-1], downsample_image)

            pySISF.sisf.create_shard(
                out_file_data.replace(".data", f".{scale}X.data"),
                out_file_meta.replace(".meta", f".{scale}X.meta"),
                downsample_image,
                (32, 32, 10),
                1,
                crop=crop,
                thread_count=8,
            )

            downsample_pyramid.append(downsample_image)

    del downsample_pyramid

print("Total Time:", time.time() - start)
