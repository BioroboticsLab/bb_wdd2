import click
import cv2
import sys
import time
import numpy as np

from multiprocessing_generator import ParallelGenerator
from skimage.transform import resize

from wdd.camera import OpenCVCapture, Flea3Capture, cam_generator
from wdd.processing import FrequencyDetector, WaggleDetector
from wdd.export import WaggleExporter


@click.command()
@click.option('--capture_type', default='OpenCV', help='Whether to use OpenCV or PyCapture2 to aquire images')
@click.option('--video_device', required=True, help='OpenCV video device. Can be camera index or video path')
@click.option('--height', default=180, help='Video frame height in px')
@click.option('--width', default=341, help='Video frame width in px')
@click.option('--fps', default=60, help='Frames per second')
@click.option('--bee_length', default=20, help='Approximate length of a bee in px')
@click.option('--binarization_threshold', default=3.25, help='Binarization threshold for waggle detection in log scale. Can be used to tune sensitivity/specitivity')
@click.option('--max_frame_distance', default=0.2, help='Maximum time inbetween frequency detections within one waggle in seconds')
@click.option('--min_num_detections', default=0.2, help='Minimum time of a waggle in seconds')
@click.option('--output_path', type=click.Path(exists=True), required=True, help='Output path for results.')
@click.option('--cam_identifier', required=True, help='Identifier of camera (used in output storage path).')
@click.option('--debug', default=False, help='Enable debug outputs/visualization')
def main(capture_type, video_device, height, width, fps, bee_length, binarization_threshold, max_frame_distance, min_num_detections, output_path, cam_identifier, debug):
    max_distance = bee_length / 4
    binarization_threshold = np.expm1(binarization_threshold)
    max_frame_distance = max_frame_distance * fps
    min_num_detections = min_num_detections * fps
    
    if capture_type == 'OpenCV':
        cam_obj = OpenCVCapture
    elif capture_type == 'PyCapture2':
        cam_obj = Flea3Capture
    else:
        raise RuntimeError('capture_type must be either OpenCV or PyCapture2')
    
    # TODO Ben: respect original video size
    full_frame_buffer_roi_size = 50
    pad_size = full_frame_buffer_roi_size // 2
    full_frame_buffer_len = 100
    full_frame_buffer = np.zeros((full_frame_buffer_len, 360 + 2 * pad_size, 683 + 2 * pad_size), dtype=np.uint8)

    dd = FrequencyDetector(height=height, width=width, fps=fps)
    exporter = WaggleExporter(cam_id=cam_identifier, output_path=output_path, full_frame_buffer=full_frame_buffer,
                              full_frame_buffer_len=full_frame_buffer_len, full_frame_buffer_roi_size=full_frame_buffer_roi_size)
    wd = WaggleDetector(max_distance=max_distance, binarization_threshold=binarization_threshold, 
                        max_frame_distance=max_frame_distance, min_num_detections=min_num_detections,
                        dilation_selem_radius=5, exporter=exporter)

    frame_idx = 0
    start_time = time.time()
    with ParallelGenerator(cam_generator(cam_obj, width=width, height=height, fps=fps, device=video_device)) as gen:
        for ret, frame, frame_orig, background in gen:
            if frame_idx % 1000 == 0:
                start_time = time.time()
    
            #ret, frame, frame_orig = cam.get_frame()
            
            if not ret:
                print('Unable to retrieve frame from video device')
                break
    
            full_frame_buffer[frame_idx % full_frame_buffer_len, pad_size:-pad_size, pad_size:-pad_size] = frame_orig.copy()
    
            r = dd.process(frame, background)
            if r is not None:
                activity, frame_diff = r
                wd.process(frame_idx, activity)
            
            # TODO Ben: Make relative to video size
            if debug and frame_idx % 101 == 0:
                current_waggle_num_detections = [len(w.xs) for w in wd.current_waggles]
                current_waggle_positions = [(w.ys[-1], w.xs[-1]) for w in wd.current_waggles]
                for blob_index, ((y, x), nd) in enumerate(zip(current_waggle_positions, current_waggle_num_detections)):
                    cv2.circle(frame_orig, (int(x * 2), int(y * 2)), 10, (0, 0, 255), 2)
                        
                """
                current_waggle_num_detections = [len(w.xs) for w in wd.finalized_waggles]
                current_waggle_positions = [(w.ys[-1], w.xs[-1]) for w in wd.finalized_waggles]
                for blob_index, ((y, x), nd) in enumerate(zip(current_waggle_positions, current_waggle_num_detections)):
                    cv2.circle(frame_orig, (int(x * 2), int(y * 2)), 10, (255, 255, 255), 2)
                """
    
                cv2.imshow('Image', resize(frame_orig, (360 * 2, 682 * 2)))
                cv2.waitKey(1)
            
            end_time = time.time()
            processing_fps = ((frame_idx % 1000) + 1) / (end_time - start_time)
            frame_idx = (frame_idx + 1)
            
            sys.stdout.write('\rCurrently processing with {:.1f} fps '.format(processing_fps))


if __name__ == '__main__':
    main()

