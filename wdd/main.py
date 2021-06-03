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


def run_wdd(
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
    debug,
    debug_frames,
    no_multiprocessing,
    no_warmup,
    start_timestamp,
    verbose=True,
    exporter_save_data_fn=None
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
        max_distance = max(2, max_distance / subsample)
    
    frame_generator = cam_generator(
        cam_obj,
        warmup=False,
        width=width,
        height=height,
        subsample=subsample,
        fps=fps,
        device=video_device,
        fullframe_path=None,
        cam_identifier=cam_identifier,
    )
    _, _, frame_orig, _ = next(frame_generator)

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
        subsampling_factor=subsample,
        save_data_fn=exporter_save_data_fn
    )
    wd = WaggleDetector(
        max_distance=max_distance,
        binarization_threshold=binarization_threshold,
        max_frame_distance=max_frame_distance,
        min_num_detections=min_num_detections,
        opening_selem_radius=2,
        dilation_selem_radius=7,
        datetime_buffer=datetime_buffer,
        full_frame_buffer_len=full_frame_buffer_len,
        exporter=exporter,
    )

    fullframe_path = os.path.join(output_path, "fullframes")
    if not os.path.exists(fullframe_path):
        os.mkdir(fullframe_path)

    frame_generator = cam_generator(
        cam_obj,
        warmup=not no_warmup,
        width=width,
        height=height,
        subsample=subsample,
        fps=fps,
        device=video_device,
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
    
    processing_fps = None

    with generator_context as gen:
        for ret, frame, frame_orig, timestamp in gen:
            if frame_idx % 10000 == 0:
                start_time = time.time()

            if not ret:
                print("Unable to retrieve frame from video device")
                break

            full_frame_buffer[
                frame_idx % full_frame_buffer_len, pad_size:-pad_size, pad_size:-pad_size
            ] = frame_orig
            datetime_buffer[frame_idx % full_frame_buffer_len] = timestamp

            activity = dd.process(frame)
            if activity is not None:
                wd.process(frame_idx, activity)

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

                im = (frame_orig * 255).astype(np.uint8)
                im = np.repeat(im[:, :, None], 3, axis=2)
                activity_im = (activity - activity.min())
                activity_im /= activity_im.max()
                activity_im = (activity_im * 255.0).astype(np.uint8)
                activity_im = cv2.applyColorMap(activity_im, cv2.COLORMAP_VIRIDIS)
                im = cv2.addWeighted(im, 0.25, activity_im, 0.75, 0)
                cv2.imshow("WDD", im)
                cv2.waitKey(1)

            if frame_idx > 0 and (frame_idx % fps == 0) and verbose:
                end_time = time.time()
                processing_fps = ((frame_idx % 10000) + 1) / (end_time - start_time)
                sys.stdout.write(
                    "\rCurrently processing with FPS: {:.1f} | Max DD: {:.2f} | [{:16s} {}]".format(
                        processing_fps, np.log1p(activity.max()), cam_identifier, video_device
                    )
                )
                sys.stdout.flush()

            frame_idx = frame_idx + 1

    return processing_fps
