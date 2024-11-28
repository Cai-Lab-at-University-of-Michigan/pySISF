#   ---------------------------------------------------------------------------------
#   Copyright (c) University of Michigan 2020-2025. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------

from enum import Enum
import numpy as np
import subprocess

ffmpeg_exe = "ffmpeg"

EncoderType = Enum("EncoderType", ["X264", "X265", "AV1_AOM", "AV1_SVT"])


def encode_stack(input_stack, method=EncoderType.X264, debug=False, fps=24):
    t = input_stack.shape[0]
    w = input_stack.shape[1]
    h = input_stack.shape[2]

    ffmpeg_command = [
        ffmpeg_exe,
        # Formatting for the input stream
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-s",
        f"{h}x{w}",
        "-r",
        f"{fps}/1",
        "-i",
        "-",
        # Formatting for the output stream
        "-an",
        "-f",
        "rawvideo",
        "-r",
        f"{fps}/1",
        "-pix_fmt",
        "gray",
        "-vcodec",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17"
        # Codec and output location added below
    ]

    match method:
        case EncoderType.X264:
            ffmpeg_command.append("-vcodec")
            ffmpeg_command.append("libx264")
        case EncoderType.X265:
            ffmpeg_command.append("-vcodec")
            ffmpeg_command.append("libx265")
        case EncoderType.AV1_AOM:
            ffmpeg_command.append("-vcodec")
            ffmpeg_command.append("libaom-av1")
        case EncoderType.AV1_SVT:
            ffmpeg_command.append("-vcodec")
            ffmpeg_command.append("libsvtav1")
        case _:
            raise ValueError(f"Unknown method {method}.")

    ffmpeg_command.append("pipe:")

    job = subprocess.Popen(
        ffmpeg_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        # stderr = subprocess.PIPE
    )

    match input_stack.dtype:
        case np.uint8:
            to_encoder = input_stack.tobytes()
        case np.uint16:
            # apply rescale...
            to_encoder = np.array(input_stack, dtype=float)
            to_encoder /= to_encoder.max()
            to_encoder *= 2**8
            to_encoder = to_encoder.astype(np.uint8).tobytes()
        case _:
            raise ValueError(f"Invalid data input type {input_stack.dtype}.")

    out, err = job.communicate(input=to_encoder)

    if not len(out):
        raise ValueError("No output receieved from ffmpeg. Is your chunk size sufficient?")

    return out


def decode_stack(input_blob, dims=(128, 128), method="libx264", debug=False, fps='24/1'):
    ffmpeg_command = [
        ffmpeg_exe,
        # Formatting for the input stream
        "-r",
        fps,
        "-i",
        "pipe:",
        # Formatting for the output stream
        "-an",
        "-f",
        "rawvideo",
        "-r",
        fps,
        "-pix_fmt",
        "gray",
        "-vcodec",
        "rawvideo",
        # Codec and output location added below
        "pipe:"
    ]

    job = subprocess.Popen(
        ffmpeg_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        # stderr = subprocess.PIPE
    )

    to_encoder = input_blob

    out, err = job.communicate(input=to_encoder)

    out_np = np.frombuffer(out, dtype=np.uint8)

    t_size = out_np.shape[0] // (dims[0] * dims[1])
    out_np = out_np.reshape((t_size, *dims))

    return out_np
