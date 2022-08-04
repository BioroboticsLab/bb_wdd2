import click
import datetime
import os
import sys
import time
import math
import numpy as np

from wdd.main import run_wdd

def show_default_option(*args, **kwargs):
    return click.option(show_default=True, *args, **kwargs)


@click.command()
@click.option(
    "--capture_type",
    default="OpenCV",
    type=click.Choice(["OpenCV", "PyCapture2", "PySpin"], case_sensitive=False),
    help="Which interface to use to acquire images. For video files, use OpenCV.",
)
@click.option(
    "--video_device", required=True, help="OpenCV video device. Can be camera index or video path"
)
@click.option(
    "--video_device_fourcc", default=None, help="OpenCV fourcc code (e.g. 'MJPG'). Depends on the camera. Only used with capture_type 'OpenCV'."
)
@click.option(
    "--video_device_api", default="CAP_ANY", help="OpenCV Capture API backend. Only used with capture_type 'OpenCV'."
)
@show_default_option("--height", default=180, help="Video frame height in px (before subsampling).")
@show_default_option("--width", default=342, help="Video frame width in px (before subsampling).")
@click.option("--subsample", default=0, help="Fast subsampling by using every Xth pixel of the images.")
@show_default_option("--fps", default=60, help="Frames per second")
@show_default_option("--bee_length", default=7, help="Approximate length of a bee in px (before subsampling).")
@show_default_option(
    "--binarization_threshold",
    default=6.0,
    help="Binarization threshold for waggle detection in log scale. Can be used to tune sensitivity/specitivity",
)
@show_default_option(
    "--max_frame_distance",
    default=0.5,
    help="Maximum time inbetween frequency detections within one waggle in seconds",
)
@show_default_option(
    "--min_num_detections", default=0.1, help="Minimum time of a waggle in seconds"
)
@click.option(
    "--output_path", type=click.Path(exists=True), required=True, help="Output path for results."
)
@click.option(
    "--cam_identifier", required=True, help="Identifier of camera (used in output storage path)."
)
@show_default_option("--debug", is_flag=True, help="Enable debug outputs/visualization")
@show_default_option(
    "--debug_frames",
    default=11,
    help="Only visualize every debug_frames frame in debug mode (can be slow if low)",
)
@click.option(
    "--use_multiprocessing",
    is_flag=True,
    help="Use a multiprocessing queue to fetch the images."
)
@click.option(
    "--no_warmup",
    is_flag=True,
    help="Do not warm up the image retrieval.."
)
@click.option(
    "--start_timestamp",
    default=None,
    help="Instead of using the wall-clock, generate camera timestamps based on the FPS starting at this iso-formatted timestamp (example: '2019-08-30T12:30:05.000100+00:00')."
)
@click.option(
    "--autoopt",
    default="",
    help="Automatically optimize hyperparameters given a CSV file with annotations."
)
@click.option(
    "--eval",
    default="",
    help="Check the detected waggles against a given CSV file and print evaluation results."
)
@click.option(
    "--roi",
    nargs=4,
    type=int,
    default=None,
    help="Specify a region of interest in pixels in the original video. The four arguments are 'left, top, width, height'."
)
@click.option(
    "--ipc",
    type=str,
    default=None,
    help="Socket address to send out detections to (e.g. 'localhost:9901:password')."
)
@click.option(
    "--record_video",
    type=str,
    default=None,
    help="Specify the OpenCV FourCC code to record a video to instead of processing (e.g. 'HFYU' or 'png ')."
)
@click.option(
    "--no_fullframes",
    is_flag=True,
    help="Do not save full-sized images in regular intervals."
)
@click.option(
    "--filter_model_path",
    type=str,
    default=None,
    help="Path to the bb_wdd_filter pytorch model checkpoint. If given, the convolutional neural network is used as the decoding stage."
)
@click.option(
    "--no_saving",
    is_flag=True,
    help="Do not serialize detections to the file system."
)
@click.option(
    "--save_waggles_only",
    is_flag=True,
    help="Needs the filter model. Disregards all detections not classified as 'waggle'."
)
@click.option(
    "--rtmp_stream_address",
    type=str,
    help="If supplied, all frames will be streamed to a RTMP address (e.g. YT). Mutually exclusive with recording to local file."
)
def main(
    bee_length,
    autoopt,
    **kwargs
):
    if not autoopt:
        run_wdd(bee_length=bee_length, **kwargs)
        print("\nStopping.")
    else:
        print("Optimizing hyperparameters..")
        from wdd.autoopt import run_optimization
        run_optimization(bee_length=bee_length, gt_path=autoopt, **kwargs)

        



if __name__ == "__main__":
    main()
