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

from .torch_support import torch

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
    use_multiprocessing,
    no_warmup,
    start_timestamp,
    stop_timestamp,
    roi,
    verbose=True,
    quiet=False,
    export_steps=None,
    eval="",
    ipc=None,
    record_video=None,
    no_fullframes=False,
    filter_model_path=None,
    no_saving=False,
    save_apngs=False,
    save_waggles_only=False,
    save_only_subset=None,
    video_device_fourcc=None,
    video_device_api=None,
    rtmp_stream_address=None,
    stream_fps=None,
    no_processing=False,
    stop_processing_on_low_fps=False,
    stream_codec=None
):
    # FIXME: should be proportional to fps (how fast can a bee move in one frame while dancing)
    max_distance = bee_length
    
    binarization_threshold = np.expm1(binarization_threshold)
    max_frame_distance = max_frame_distance * fps
    min_num_detections = min_num_detections * fps

    capture_dependent_camera_kwargs = dict()
    if capture_type == "OpenCV":
        cam_obj = OpenCVCapture
        capture_dependent_camera_kwargs["fourcc"] = video_device_fourcc
        capture_dependent_camera_kwargs["capture_api"] = video_device_api
    elif capture_type == "PyCapture2":
        cam_obj = Flea3Capture
    elif capture_type == "PySpin":
        cam_obj = Flea3CapturePySpin
    else:
        raise RuntimeError("capture_type must be either OpenCV, PyCapture2, PySpin")

    image_input_width = width
    image_input_height = height

    roi_offset_x, roi_offset_y = 0, 0
    if roi is not None and roi:
        width = int(roi[2])
        height = int(roi[3])
        roi_offset_x, roi_offset_y = int(roi[0]), int(roi[1])
    else:
        roi = None

    video_writer = None
    enable_processing = not no_processing
    record_input, record_output = False, False
    if record_video:
        print("Recording to file in current working directory instead of processing (codec = '{}')...".format(record_video))
        writer_fps = fps
        if not debug: # In non-debug mode, recording to file skips processing.
            enable_processing = False
        else:
            writer_fps //= debug_frames
        video_writer = VideoWriter(str(video_device), writer_fps, record_video, directory=output_path)
    elif rtmp_stream_address:
        print("Streaming instead of processing.")
        from wdd.streamer import RTMPStreamer
        video_writer = RTMPStreamer(rtmp_stream_address, input_fps=fps, output_fps=stream_fps, debug=debug, stream_codec=stream_codec)

    if video_writer is not None:
        if debug:
            record_output = True
        else:
            record_input = True

    # Compose a small string that describes what we do.
    if verbose:
        what = []
        if enable_processing:
            what.append("processing")
        if rtmp_stream_address is not None:
            what.append("streaming")
        elif video_writer is not None:
            what.append("recording")
        processing_label = "+".join(what)

    subsample = int(subsample)
    if subsample > 1:
        height = math.ceil(height / subsample)
        width = math.ceil(width / subsample)
        max_distance = max(2, max_distance / subsample)

    frame_generator = cam_generator(
        cam_obj,
        warmup=False,
        width=image_input_width,
        height=image_input_height,
        target_width=width,
        target_height=height,
        subsample=subsample,
        fps=fps,
        device=video_device,
        fullframe_path=None,
        cam_identifier=cam_identifier,
        roi=roi,
        quiet=quiet,
        **capture_dependent_camera_kwargs
    )
    _, _, frame_orig, _ = next(frame_generator)
    # Clean up camera object.
    del frame_generator

    full_frame_buffer_roi_size = bee_length * 5
    full_frame_buffer_len = 3 * int(fps)
    # Buffer is a list instead of an array to save one copy if no images need to be saved.
    full_frame_buffer = [None] * full_frame_buffer_len
    datetime_buffer = [datetime.datetime.min.isoformat() for _ in range(full_frame_buffer_len)]

    waggle_decoder = None
    class_filter = []
    if not filter_model_path:
        from wdd.decoding import WaggleDecoder
        waggle_decoder = WaggleDecoder(fps=fps, bee_length=bee_length)
    else:
        from wdd.decoding_convnet import WaggleDecoderConvNet
        waggle_decoder = WaggleDecoderConvNet(fps=fps, bee_length=bee_length,
                model_path=filter_model_path)
        if save_waggles_only or save_only_subset is not None:
            include_classes = ["waggle"] if save_waggles_only else None

            class_filter = [ClassFilter(include_classes=include_classes,
                                        save_only_subset=save_only_subset)]

    dd = FrequencyDetector(height=height, width=width, fps=fps)
    waggle_metadata_saver, external_interface = None, None
    waggle_serializer = None
    if export_steps is None:
        export_steps = [waggle_decoder] + class_filter
        if ipc:
            from wdd.remote_interface import ResultsSender
            external_interface = ResultsSender(address_string=ipc)
            export_steps.append(external_interface)
        if eval or debug:
            from wdd.evaluation import WaggleMetadataSaver
            waggle_metadata_saver = WaggleMetadataSaver()
            export_steps.append(waggle_metadata_saver)

        if not no_saving:
            waggle_serializer = WaggleSerializer(cam_id=cam_identifier, output_path=output_path,
                                                save_apngs=save_apngs, fps=fps)
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
        quiet=quiet,
        export_steps=export_steps
    )
    wd = WaggleDetector(
        max_distance=max_distance,
        binarization_threshold=binarization_threshold,
        max_frame_distance=max_frame_distance,
        min_num_detections=min_num_detections,
        opening_selem_size=3,
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
        width=image_input_width,
        height=image_input_height,
        target_width=width,
        target_height=height,
        subsample=subsample,
        fps=fps,
        device=video_device,
        fullframe_path=fullframe_path if not no_fullframes else "",
        cam_identifier=cam_identifier,
        start_timestamp=start_timestamp,
        stop_timestamp=stop_timestamp,
        roi=roi,
        quiet=quiet,
        **capture_dependent_camera_kwargs
    )

    frame_idx = 0
    start_time = time.time()

    generator_context = None
    if use_multiprocessing:
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
                
                if enable_processing:

                    full_frame_buffer[frame_idx % full_frame_buffer_len] = frame_orig
                    datetime_buffer[frame_idx % full_frame_buffer_len] = timestamp

                    activity = dd.process(frame)
                    if activity is not None:
                        wd.process(frame_idx, activity)

                    if debug and frame_idx % debug_frames == 0:
                        im = frame_orig.copy()
                        
                        if im.max() < 1.0 + 1e-5:
                            im = im * 255.0
                        im = im.astype(np.uint8)
                        if len(im.shape) == 2:
                            im = np.repeat(im[:, :, None], 3, axis=2)
                        
                        # Scale image range robustly.
                        activity_im = activity

                        if torch is not None:
                            activity_im = activity_im.cpu().numpy()

                        activity_min = activity_im.min()
                        activity_max = activity_im.max()

                        if activity_norm[0] > activity_min:
                            activity_norm[0] = activity_min
                        else:
                            activity_norm[0] = (0.9 * activity_norm[0]) + 0.1 * activity_min
                        activity_im = (activity_im - activity_norm[0])
                        
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
                            padding_height = (roi[1], im.shape[0] - h - roi[1])
                            padding_width = (roi[0], im.shape[1] - w - roi[0])
                            if padding_height[1] < 0 or padding_width[1] < 0:
                                raise ValueError("Invalid ROI given. Please check size. (ROI: {}, Image size: {})".format(
                                    roi, im.shape))
                            activity_im = np.pad(activity_im, (padding_height, padding_width))
                        activity_im = (activity_im * 255.0).astype(np.uint8)
                        activity_im = cv2.applyColorMap(activity_im, cv2.COLORMAP_VIRIDIS)
                        # Due to rounding, the activity image can be a few pixels larger than the 'normal' image now.
                        activity_im = activity_im[0:im.shape[0], 0:im.shape[1], :]
                        im = cv2.addWeighted(im, 0.25, activity_im, 0.75, 0)

                        current_waggle_num_detections = [len(w.xs) for w in wd.current_waggles]
                        current_waggle_positions = [(w.ys[-1], w.xs[-1]) for w in wd.current_waggles]
                        for blob_index, ((y, x), nd) in enumerate(
                            zip(current_waggle_positions, current_waggle_num_detections)
                        ):
                            cv2.circle(
                                im,
                                (int(x * subsample + roi_offset_x), int(y * subsample + roi_offset_y)),
                                10,
                                (0, 0, 255),
                                2,
                            )
                        
                        if waggle_metadata_saver is not None:
                            finished_waggles = waggle_metadata_saver.all_waggles[-2:]
                            for waggle_metadata in finished_waggles:
                                x_pos = np.median(waggle_metadata["x_coordinates"])
                                y_pos = np.median(waggle_metadata["y_coordinates"])
                                class_label = waggle_metadata.get("predicted_class_label", "waggle")

                                if class_label != "waggle":
                                    colormap = dict(activating=(0, 255, 100), ventilating=(255, 150, 150), other=(50, 50, 50))
                                    color = colormap.get(class_label, (200, 200, 200))
                                    radius = bee_length // 2
                                    line_width = 2

                                    if class_label == "other":
                                        radius //= 2
                                        line_width = 1

                                    cv2.circle(
                                        im,
                                        (int(x_pos), int(y_pos)),
                                        radius,
                                        color,
                                        line_width,
                                    )
                                else:
                                    angle = waggle_metadata["waggle_angle"]
                                    direction_x, direction_y = bee_length * np.cos(angle), -bee_length * np.sin(angle)
                                    cv2.arrowedLine(im, (int(x_pos - direction_x), int(y_pos - direction_y)),
                                                (int(x_pos + direction_x), int(y_pos + direction_y)),
                                                (0, 255, 255), 1)

                        if record_output:
                            video_writer.write(im)
                        else:
                            cv2.imshow("WDD", im[::2,::2])
                            cv2.waitKey(1)

                if frame_idx > 0 and (frame_idx % fps == 0):
                    end_time = time.time()
                    processing_fps = ((frame_idx % 10000) + 1) / (end_time - start_time)

                    if stop_processing_on_low_fps and (processing_fps < fps / 2) and (frame_idx > fps * 2):
                        print("\rStopping due to low FPS ({:3.2f} FPS).".format(processing_fps))
                        break

                    if verbose:

                        max_activity = activity.max()
                        if torch is not None and not activity.ndim == 1:
                            max_activity = float(max_activity.cpu())

                        sys.stdout.write(
                            "\r{}, FPS: {:.1f} | Max DD: {:.2f} | [{}, {}]".format(
                                processing_label, processing_fps, np.log1p(max_activity), cam_identifier, video_device
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
        if not quiet:
            print("\nSaving running exports..", flush=True)
        exporter.finalize_exports()
        if waggle_serializer is not None:
            waggle_serializer.finalize_serialization()

    if eval and (waggle_metadata_saver is not None):
        import wdd.evaluation
        ground_truth = wdd.evaluation.load_ground_truth(eval, video_path=video_device, fps=fps, start_timestamp=first_timestamp)
        results = wdd.evaluation.calculate_scores(waggle_metadata_saver.all_waggles, ground_truth, bee_length=bee_length, verbose=False)
        print(results)
    return processing_fps
