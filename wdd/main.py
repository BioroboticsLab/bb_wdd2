import click
import cv2
import datetime
import os
import sys
import skimage.transform
import time
import math
import numpy as np

from wdd.camera import OpenCVCapture, Flea3Capture, Flea3CapturePySpin, cam_generator
from wdd.processing import FrequencyDetector, WaggleDetector
from wdd.export import WaggleSerializer, WaggleExportPipeline, VideoWriter, ClassFilter

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
    roi,
    verbose=True,
    export_steps=None,
    eval="",
    ipc=None,
    record_video=None,
    no_fullframes=False,
    filter_model_path=None,
    no_saving=False,
    save_waggles_only=False
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
    elif capture_type == "PySpin":
        cam_obj = Flea3CapturePySpin
    else:
        raise RuntimeError("capture_type must be either OpenCV, PyCapture2, PySpin")

    if roi is not None and roi:
        width = int(roi[2])
        height = int(roi[3])
    else:
        roi = None

    video_writer = None
    record_input, record_output = False, False
    if record_video:
        print("Recording to file in current working directory instead of processing (codec = '{}')...".format(record_video))
        video_writer = VideoWriter(str(video_device), fps, record_video)
        if debug:
            record_output = True
        else:
            record_input = True

    subsample = int(subsample)
    if subsample > 1:
        height = math.ceil(height / subsample)
        width = math.ceil(width / subsample)
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
        roi=roi
    )
    _, _, frame_orig, _ = next(frame_generator)

    full_frame_buffer_roi_size = bee_length * 5
    full_frame_buffer_len = 3 * int(fps)
    # Buffer is a list instead of an array to save one copy if no images need to be saved.
    full_frame_buffer = [None] * full_frame_buffer_len
    datetime_buffer = [datetime.datetime.min.isoformat() for _ in range(full_frame_buffer_len)]

    frame_scale = frame_orig.shape[0] / height, frame_orig.shape[1] / width

    waggle_decoder = None
    class_filter = []
    if not filter_model_path:
        from wdd.decoding import WaggleDecoder
        waggle_decoder = WaggleDecoder(fps=fps, bee_length=bee_length)
    else:
        from wdd.decoding_convnet import WaggleDecoderConvNet
        waggle_decoder = WaggleDecoderConvNet(fps=fps, bee_length=bee_length,
                model_path=filter_model_path)
        if save_waggles_only:
            class_filter = [ClassFilter(include_classes=["waggle"])]

    dd = FrequencyDetector(height=height, width=width, fps=fps)
    waggle_metadata_saver, external_interface = None, None
    waggle_serializer = None
    if export_steps is None:
        export_steps = [waggle_decoder] + class_filter
        if ipc:
            from wdd.remote_interface import ResultsSender
            external_interface = ResultsSender(address_string=ipc)
            export_steps.append(external_interface)
        if eval:
            from wdd.evaluation import WaggleMetadataSaver
            waggle_metadata_saver = WaggleMetadataSaver()
            export_steps.append(waggle_metadata_saver)

        if not no_saving:
            waggle_serializer = WaggleSerializer(cam_id=cam_identifier, output_path=output_path)
            export_steps.append(waggle_serializer)

    exporter = WaggleExportPipeline(
        cam_id=cam_identifier,
        datetime_buffer=datetime_buffer,
        full_frame_buffer=full_frame_buffer,
        full_frame_buffer_len=full_frame_buffer_len,
        full_frame_buffer_roi_size=full_frame_buffer_roi_size,
        subsampling_factor=subsample,
        fps=fps,
        min_images=int(1.5 * fps),
        roi=roi,
        export_steps=export_steps
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
        fullframe_path=fullframe_path if not no_fullframes else "",
        cam_identifier=cam_identifier,
        start_timestamp=start_timestamp,
        roi=roi
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
    activity = np.array([np.nan])
    activity_norm = [+np.inf, -np.inf]
    first_timestamp = None
    try:
            
        with generator_context as gen:
            for ret, frame, frame_orig, timestamp in gen:
                if frame_idx % 10000 == 0:
                    start_time = time.time()
                if first_timestamp is None:
                    first_timestamp = timestamp
                if not ret:
                    print("Unable to retrieve frame from video device")
                    break

                if record_input:
                    video_writer.write(frame_orig)
                else:

                    full_frame_buffer[frame_idx % full_frame_buffer_len] = frame_orig
                    datetime_buffer[frame_idx % full_frame_buffer_len] = timestamp

                    activity = dd.process(frame)
                    if activity is not None:
                        wd.process(frame_idx, activity)

                    if debug and frame_idx % debug_frames == 0:
                        im = frame_orig.copy()
                        current_waggle_num_detections = [len(w.xs) for w in wd.current_waggles]
                        current_waggle_positions = [(w.ys[-1], w.xs[-1]) for w in wd.current_waggles]
                        for blob_index, ((y, x), nd) in enumerate(
                            zip(current_waggle_positions, current_waggle_num_detections)
                        ):
                            offset_x, offset_y = 0, 0
                            if roi is not None:
                                offset_x, offset_y = roi[:2]
                            cv2.circle(
                                im,
                                (int(x * frame_scale[0] + offset_x), int(y * frame_scale[1] + offset_y)),
                                10,
                                (0, 0, 255),
                                2,
                            )
                        
                        if im.max() < 1.0 + 1e-5:
                            im = im * 255.0
                        im = im.astype(np.uint8)
                        if len(im.shape) == 2:
                            im = np.repeat(im[:, :, None], 3, axis=2)
                        # Scale image range robustly.
                        activity_min = activity.min()
                        if activity_norm[0] > activity_min:
                            activity_norm[0] = activity_min
                        else:
                            activity_norm[0] = (0.9 * activity_norm[0]) + 0.1 * activity_min
                        activity_im = (activity - activity_norm[0])

                        activity_max = activity.max()
                        if activity_norm[1] < activity_max:
                            activity_norm[1] = activity_max
                        else:
                            activity_norm[1] = (0.9 * activity_norm[1]) + 0.1 * activity_max
                        activity_im /= activity_norm[1]
                        
                        h, w = height, width
                        if subsample > 1:
                            h, w = h * subsample, w * subsample
                        activity_im = skimage.transform.resize(activity_im, (h, w))
                        if roi is not None:
                            activity_im = np.pad(activity_im,
                                        ((roi[1], im.shape[0] - h - roi[1]),
                                        (roi[0], im.shape[1] - w - roi[0])))
                        activity_im = (activity_im * 255.0).astype(np.uint8)
                        activity_im = cv2.applyColorMap(activity_im, cv2.COLORMAP_VIRIDIS)
                        # Due to rounding, the activity image can be a few pixels larger than the 'normal' image now.
                        activity_im = activity_im[0:im.shape[0], 0:im.shape[1], :]
                        im = cv2.addWeighted(im, 0.25, activity_im, 0.75, 0)

                        if record_output:
                            video_writer.write(im)
                        else:
                            cv2.imshow("WDD", im)
                            cv2.waitKey(1)

                if frame_idx > 0 and (frame_idx % fps == 0):
                    end_time = time.time()
                    processing_fps = ((frame_idx % 10000) + 1) / (end_time - start_time)
                    if verbose:
                        what = "processing" if not video_writer else "recording"
                        sys.stdout.write(
                            "\rCurrently {} with FPS: {:.1f} | Max DD: {:.2f} | [{:16s} {}]".format(
                                what, processing_fps, np.log1p(activity.max()), cam_identifier, video_device
                            )
                        )
                        sys.stdout.flush()

                frame_idx = frame_idx + 1
    finally:
        if external_interface is not None:
            external_interface.close()
        if video_writer is not None:
            video_writer.close()
        # Wait for all remaining data to be written.
        print("\nSaving running exports..", flush=True)
        exporter.finalize_exports()
        if waggle_serializer is not None:
            waggle_serializer.finalize_serialization()

    if waggle_metadata_saver is not None:
        import wdd.evaluation
        ground_truth = wdd.evaluation.load_ground_truth(eval, video_path=video_device, fps=fps, start_timestamp=first_timestamp)
        results = wdd.evaluation.calculate_scores(waggle_metadata_saver.all_waggles, ground_truth, bee_length=bee_length, verbose=False)
        print(results)
    return processing_fps
