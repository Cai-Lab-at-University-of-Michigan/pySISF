#   ---------------------------------------------------------------------------------
#   Copyright (c) University of Michigan 2020-2024. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------

from enum import Enum

import ffmpeg
import numpy as np

fmpeg_exe = "ffmpeg"

EncoderType = Enum("EncoderType", ["X264", "X265", "AV1_AOM", "AV1_SVT"])


def encode_stack(input_stack, method=EncoderType.X264, debug=False):
    t = input_stack.shape[0]
    w = input_stack.shape[1]
    h = input_stack.shape[2]

    nodes = ffmpeg.input("pipe:", format="rawvideo", pix_fmt="gray", s=f"{h}x{w}", framerate="24/1")

    match method:
        case EncoderType.X264:
            nodes = nodes.output(
                "pipe:",
                format="rawvideo",
                pix_fmt="gray",
                framerate="24/1",
                vcodec="libx264",
            )
        case EncoderType.X265:
            nodes = nodes.output(
                "pipe:",
                format="rawvideo",
                pix_fmt="gray",
                framerate="24/1",
                vcodec="libx265",
            )
        case EncoderType.AV1_AOM:
            nodes = nodes.output(
                "pipe:",
                format="rawvideo",
                pix_fmt="gray",
                framerate="24/1",
                vcodec="libaom-av1",
            )
        case EncoderType.AV1_SVT:
            nodes = nodes.output(
                "pipe:",
                format="rawvideo",
                pix_fmt="gray",
                framerate="24/1",
                vcodec="libsvtav1",
            )
        case _:
            raise ValueError(f"Unknown method {method}.")

    process = nodes.run_async(
        cmd=fmpeg_exe if method != EncoderType.AV1_SVT else "./ffmpeg_forav1",
        pipe_stdout=True,
        pipe_stdin=True,
        pipe_stderr=(not debug),
    )

    match input_stack.dtype:
        case np.uint8:
            to_encoder = input_stack.tobytes()
        case np.uint16:
            # apply rescale...
            to_encoder = np.array(input_stack, dtype=float)
            to_encoder /= to_encoder.max()
            to_encoder *= 2**8
            to_encoder = to_encoder.tobytes()
        case _:
            raise ValueError(f"Invalid data input type {input_stack.dtype}.")

    out, err = process.communicate(input=to_encoder)

    if not len(out):
        raise ValueError("No output receieved from ffmpeg. Is your chunk size sufficient?")

    return out


def decode_stack(input_blob, dims=(128, 128), method="libx264", debug=False):
    nodes = ffmpeg.input("pipe:", framerate="24/1").output("pipe:", format="rawvideo", pix_fmt="gray", framerate="24/1")

    process = nodes.run_async(cmd=fmpeg_exe, pipe_stdout=True, pipe_stdin=True, pipe_stderr=(not debug))

    process.stdin.write(input_blob)
    process.stdin.close()

    r = process.stdout.read()

    out = np.frombuffer(r, dtype=np.uint8)

    t_size = out.shape[0] // (dims[0] * dims[1])
    out = out.reshape((t_size, *dims))

    return out
