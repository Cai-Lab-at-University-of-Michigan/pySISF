#   ---------------------------------------------------------------------------------
#   Copyright (c) University of Michigan 2020-2025. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------

from enum import Enum
import numpy as np
import subprocess

# Static builds can be downloaded from:
# - https://github.com/BtbN/FFmpeg-Builds/releases/tag/latest
# - https://johnvansickle.com/ffmpeg/
ffmpeg_exe = "ffmpeg"

EncoderType = Enum("EncoderType", ["X264", "X265", "AV1_AOM", "AV1_SVT"])

DEBUG=False


def encode_stack(input_stack, method=EncoderType.X264, debug=False, fps=24, compression_opts=None):
    if len(input_stack.shape) != 3:
        raise ValueError(f"Invalid input size {input_stack.shape}! (should be 3)")

    # Input XYZ formatted, using Z as time channel

    t = input_stack.shape[2]
    w = input_stack.shape[0]
    h = input_stack.shape[1]

    match method:
        case EncoderType.X264:
            crf = 17
            preset = "slow"
        case EncoderType.X265:
            crf = 17
            preset = "slow"
        case EncoderType.AV1_AOM:
            crf = 5
            preset = "3"
        case EncoderType.AV1_SVT:
            crf = 5
            preset = "3"
        case _:
            raise ValueError(f"Unknown method {method}.")

    if compression_opts:
        if "crf" in compression_opts:
            crf = compression_opts["crf"]
        if "preset" in compression_opts:
            preset = compression_opts["preset"]

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
        # "-vcodec",
        # "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
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
            ffmpeg_command.append("-rc")
            ffmpeg_command.append("0")
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
        stderr=subprocess.DEVNULL if not DEBUG else subprocess.STDOUT,
    )

    to_encoder = b""

    match input_stack.dtype:
        case np.uint8:
            to_encoder = np.moveaxis(input_stack, -1, 0).tobytes()
        case np.uint16:
            # apply rescale...
            to_encoder = np.array(input_stack, dtype=np.float32)
            # to_encoder /= to_encoder.max() if to_encoder.max() > 0 else 1
            # to_encoder *= 2**8

            to_encoder = to_encoder**0.5

            to_encoder = to_encoder.astype(np.uint8)

            to_encoder = np.moveaxis(to_encoder, -1, 0)
            to_encoder = to_encoder.tobytes()
        case _:
            raise ValueError(f"Invalid data input type {input_stack.dtype}.")

    out, err = job.communicate(input=to_encoder)

    if not len(out):
        raise ValueError("No output receieved from ffmpeg. Is your chunk size sufficient?")

    return out


def decode_stack(input_blob, dims=(128, 128), method="libx264", debug=False, fps="24/1"):
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
        "pipe:",
    ]

    job = subprocess.Popen(
        ffmpeg_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        #stderr = subprocess.PIPE
    )

    out, err = job.communicate(input=input_blob)

    out_np = np.frombuffer(out, dtype=np.uint8)

    t_size = out_np.shape[0] // (dims[0] * dims[1])
    out_np = out_np.reshape((t_size, *dims))  # Z X Y

    out_np = np.moveaxis(out_np, 0, -1)  # X Y Z

    return out_np
