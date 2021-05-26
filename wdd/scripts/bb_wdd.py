import click
import cv2
import datetime
import os
import sys
import time
import numpy as np

from wdd.camera import OpenCVCapture, Flea3Capture, cam_generator
from wdd.processing import FrequencyDetector, WaggleDetector
from wdd.export import WaggleExporter


def show_default_option(*args, **kwargs):
    return click.option(show_default=True, *args, **kwargs)


@click.command()
@show_default_option(
    "--capture_type",
    default="PyCapture2",
    help="Whether to use OpenCV or PyCapture2 to aquire images",
)
@click.option(
    "--video_device", required=True, help="OpenCV video device. Can be camera index or video path"
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
    default=0.2,
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
@click.option(
    "--background_path",
    type=click.Path(exists=True),
    required=True,
    help="Where to load/store background image.",
)
@show_default_option("--debug", default=False, help="Enable debug outputs/visualization")
@show_default_option(
    "--debug_frames",
    default=11,
    help="Only visualize every debug_frames frame in debug mode (can be slow if low)",
)
@click.option(
    "--no_background_updates",
    is_flag=True,
    help="Do not update the background image. Can be used for additional performance if the background is static."
)
@click.option(
    "--no_multiprocessing",
    is_flag=True,
    help="Do not use a multiprocessing queue to fetch the images."
)
@click.option(
    "--start_timestamp",
    default=None,
    help="Instead of using the wall-clock, generate camera timestamps based on the FPS starting at this iso-formatted timestamp (example: '2019-08-30T12:30:05.000100+00:00')."
)
def main(
    capture_type,
    video_device,
    height,
    width,
    subsample,
    fps,
    bee_length,
    binarization_threshold,
    max_frame_distance,
    min_num_detections,
    output_path,
    cam_identifier,
    background_path,
    debug,
    debug_frames,
    no_background_updates,
    no_multiprocessing,
    start_timestamp,
):
    # FIXME: should be proportional to fps (how fast can a bee move in one frame while dancing)
    max_distance = bee_length
    binarization_threshold = np.expm1(binarization_threshold)
    max_frame_distance = max_frame_distance * fps
    min_num_detections = min_num_detections * fps

    if capture_type == "OpenCV":
        cam_obj = OpenCVCapture
    elif capture_type == "PyCapture2":
        cam_obj = Flea3Capture
    else:
        raise RuntimeError("capture_type must be either OpenCV or PyCapture2")
    
    subsample = int(subsample)
    if subsample > 1:
        height = height // subsample
        width = width // subsample
        max_distance = max_distance / subsample

    frame_generator = cam_generator(
        cam_obj,
        warmup=False,
        width=width,
        height=height,
        subsample=subsample,
        fps=fps,
        device=video_device,
        background=None,
        fullframe_path=None,
        cam_identifier=cam_identifier,
    )
    _, _, frame_orig, _, _ = next(frame_generator)

    full_frame_buffer_roi_size = bee_length * 10
    pad_size = full_frame_buffer_roi_size // 2
    full_frame_buffer_len = 100
    full_frame_buffer = np.zeros(
        (
            full_frame_buffer_len,
            frame_orig.shape[0] + 2 * pad_size,
            frame_orig.shape[1] + 2 * pad_size,
        ),
        dtype=np.float32,
    )
    datetime_buffer = [datetime.datetime.min.isoformat() for _ in range(full_frame_buffer_len)]

    frame_scale = frame_orig.shape[0] / height, frame_orig.shape[1] / width

    dd = FrequencyDetector(height=height, width=width, fps=fps)
    exporter = WaggleExporter(
        cam_id=cam_identifier,
        output_path=output_path,
        datetime_buffer=datetime_buffer,
        full_frame_buffer=full_frame_buffer,
        full_frame_buffer_len=full_frame_buffer_len,
        full_frame_buffer_roi_size=full_frame_buffer_roi_size,
    )
    wd = WaggleDetector(
        max_distance=max_distance,
        binarization_threshold=binarization_threshold,
        max_frame_distance=max_frame_distance,
        min_num_detections=min_num_detections,
        dilation_selem_radius=7,
        datetime_buffer=datetime_buffer,
        full_frame_buffer_len=full_frame_buffer_len,
        exporter=exporter,
    )

    background_file = os.path.join(background_path, "background_{}.npy".format(cam_identifier))
    if os.path.exists(background_file):
        print("Loading background image: {}".format(background_file))
        background = np.load(background_file)
    else:
        print("No background image found for {}, starting from scratch".format(cam_identifier))
        background = None

    fullframe_path = os.path.join(output_path, "fullframes")
    if not os.path.exists(fullframe_path):
        os.mkdir(fullframe_path)

    frame_generator = cam_generator(
        cam_obj,
        warmup=True,
        width=width,
        height=height,
        subsample=subsample,
        fps=fps,
        device=video_device,
        background=background,
        no_background_updates=no_background_updates,
        fullframe_path=None,
        cam_identifier=cam_identifier,
        start_timestamp=start_timestamp,
    )

    frame_idx = 0
    start_time = time.time()

    generator_context = None
    if not no_multiprocessing:
        try:
            from multiprocessing_generator import ParallelGenerator
            generator_context = ParallelGenerator(frame_generator, max_lookahead=fps)
        except Exception as e:
            print("Failed to import and use multiprocessing_generator. Falling back to sequential reading. [Error: '{}']".format(str(e)))

    if generator_context is None:
        import contextlib
        generator_context = contextlib.nullcontext(frame_generator)

    with generator_context as gen:
        for ret, frame, frame_orig, background, timestamp in gen:
            if frame_idx % 10000 == 0:
                start_time = time.time()

            if not ret:
                print("Unable to retrieve frame from video device")
                break

            full_frame_buffer[
                frame_idx % full_frame_buffer_len, pad_size:-pad_size, pad_size:-pad_size
            ] = frame_orig
            datetime_buffer[frame_idx % full_frame_buffer_len] = timestamp

            r = dd.process(frame, background)
            if r is not None:
                activity, frame_diff = r
                wd.process(frame_idx, activity)

            if (frame_idx > 0) and (frame_idx % 10000 == 0) and not no_background_updates:
                print("\nSaving background image: {}".format(background_file))
                np.save(background_file, background)

            if debug and frame_idx % debug_frames == 0:
                current_waggle_num_detections = [len(w.xs) for w in wd.current_waggles]
                current_waggle_positions = [(w.ys[-1], w.xs[-1]) for w in wd.current_waggles]
                for blob_index, ((y, x), nd) in enumerate(
                    zip(current_waggle_positions, current_waggle_num_detections)
                ):
                    cv2.circle(
                        frame_orig,
                        (int(x * frame_scale[0]), int(y * frame_scale[1])),
                        10,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("WDD", (frame_orig * 255).astype(np.uint8))
                cv2.waitKey(1)

            if frame_idx % fps == 0:
                end_time = time.time()
                processing_fps = ((frame_idx % 10000) + 1) / (end_time - start_time)
                sys.stdout.write(
                    "\rCurrently processing with FPS: {:.1f} | Max DD: {:.2f} | [{:16s} {}]".format(
                        processing_fps, np.log1p(activity.max()), cam_identifier, video_device
                    )
                )
                sys.stdout.flush()

            frame_idx = frame_idx + 1
    print("\nStopping.")

    if (frame_idx > 0) and not no_background_updates:
        print("\nSaving background image: {}".format(background_file))
        np.save(background_file, background)

if __name__ == "__main__":
    main()
