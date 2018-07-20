import click
import cv2
import os
import sys
import time
import numpy as np
from imageio import imsave

from multiprocessing_generator import ParallelGenerator
from skimage.transform import resize

from wdd.camera import OpenCVCapture, Flea3Capture, cam_generator
from wdd.processing import FrequencyDetector, WaggleDetector
from wdd.export import WaggleExporter

def show_default_option(*args, **kwargs):
    return click.option(show_default=True, *args, **kwargs)

@click.command()
@show_default_option('--capture_type', default='PyCapture2', help='Whether to use OpenCV or PyCapture2 to aquire images')
@click.option('--video_device', required=True, help='OpenCV video device. Can be camera index or video path')
@show_default_option('--height', default=180, help='Video frame height in px')
@show_default_option('--width', default=342, help='Video frame width in px')
@show_default_option('--fps', default=60, help='Frames per second')
@show_default_option('--bee_length', default=7, help='Approximate length of a bee in px')
@show_default_option('--binarization_threshold', default=6., help='Binarization threshold for waggle detection in log scale. Can be used to tune sensitivity/specitivity')
@show_default_option('--max_frame_distance', default=0.2, help='Maximum time inbetween frequency detections within one waggle in seconds')
@show_default_option('--min_num_detections', default=0.1, help='Minimum time of a waggle in seconds')
@click.option('--output_path', type=click.Path(exists=True), required=True, help='Output path for results.')
@click.option('--cam_identifier', required=True, help='Identifier of camera (used in output storage path).')
@click.option('--background_path', type=click.Path(exists=True), required=True, help='Where to load/store background image.')
@show_default_option('--debug', default=False, help='Enable debug outputs/visualization')
def main(capture_type, video_device, height, width, fps, bee_length, binarization_threshold, max_frame_distance, 
         min_num_detections, output_path, cam_identifier, background_path, debug):
    # FIXME: should be proportional to fps (how fast can a bee move in one frame while dancing)
    max_distance = bee_length
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
                        dilation_selem_radius=7, exporter=exporter)

    background_file = os.path.join(background_path, 'background_{}.npy'.format(cam_identifier))
    if os.path.exists(background_file):
        print('Loading background image: {}'.format(background_file))
        background = np.load(background_file)
    else:
        print('No background image found for {}, starting from scratch'.format(cam_identifier))
        background = None

    fullframe_path = os.path.join(output_path, 'fullframes')
    if not os.path.exists(fullframe_path):
        os.mkdir(fullframe_path)

    frame_generator = cam_generator(cam_obj, width=width, height=height, fps=fps, device=video_device,
                                    background=background, fullframe_path=fullframe_path)


    frame_idx = 0
    start_time = time.time()

    with ParallelGenerator(frame_generator, max_lookahead=fps) as gen:
        for ret, frame, frame_orig, background in gen:
            if frame_idx % 10000 == 0:
                start_time = time.time()
    
            if not ret:
                print('Unable to retrieve frame from video device')
                break
    
            full_frame_buffer[frame_idx % full_frame_buffer_len, pad_size:-pad_size, pad_size:-pad_size] = \
                (((frame_orig + 1) / 2) * 255).astype(np.uint8)
    
            r = dd.process(frame, background)
            if r is not None:
                activity, frame_diff = r
                wd.process(frame_idx, activity)

            if frame_idx % 10000 == 0:
                print('\nSaving background image: {}'.format(background_file))
                np.save(background_file, background)
            
            # TODO Ben: Make relative to video size
            if debug and frame_idx % 11 == 0:
                current_waggle_num_detections = [len(w.xs) for w in wd.current_waggles]
                current_waggle_positions = [(w.ys[-1], w.xs[-1]) for w in wd.current_waggles]
                for blob_index, ((y, x), nd) in enumerate(zip(current_waggle_positions, current_waggle_num_detections)):
                    cv2.circle(frame_orig, (int(x * 2), int(y * 2)), 10, (0, 0, 255), 2)
                        
                cv2.imshow('WDD', resize((frame_orig + 1) / 2, (360 * 2, 682 * 2)))
                cv2.waitKey(1)
            
            end_time = time.time()
            processing_fps = ((frame_idx % 10000) + 1) / (end_time - start_time)
            frame_idx = (frame_idx + 1)
            
            sys.stdout.write('\rCurrently processing with FPS: {:.1f} | Max DD: {:.2f}'.format(processing_fps, np.log1p(activity.max())))


if __name__ == '__main__':
    main()

