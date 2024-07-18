#   ---------------------------------------------------------------------------------
#   Copyright (c) University of Michigan 2020-2024. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------

import glob
import tqdm
import time

import numpy as np

import pySISF.sisf
import pySISF.sndif_utils

pySISF.sisf.DEBUG = False

start = time.time()

data_files = glob.glob("data/*.1X.data")
meta_files = [x.replace("data", "meta") for x in data_files]

for dfname, mfname in tqdm.tqdm(zip(data_files, meta_files), total=len(data_files)):
    a = pySISF.sisf.sisf_chunk(dfname, mfname)

    b = np.array(a[:, :, :])

    downsample_pyramid = [b]
    for downsample_rate in range(5):
        scale = 2**downsample_rate

        if scale == 1:
            continue

        downsample_shape = tuple(i // scale for i in b.shape)
        downsample_image = np.zeros(shape=downsample_shape, dtype=np.uint16)

        pySISF.sndif_utils.downsample(downsample_pyramid[-1], downsample_image)

        pySISF.sisf.create_shard(
            dfname.replace(".1X.", f".{scale}X."),
            mfname.replace(".1X.", f".{scale}X."),
            downsample_image,
            (32, 32, 32),
            1,
            thread_count=8,
        )

        downsample_pyramid.append(downsample_image)

    del downsample_pyramid, b

print("Total Time:", time.time() - start)
