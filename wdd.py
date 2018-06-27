#!/usr/bin/env python3

import sys
import time
from collections import namedtuple

import click

import numpy as np
import cv2

import scipy
from scipy.optimize import linear_sum_assignment

from skimage.transform import resize
from skimage import measure 
from skimage import morphology
from skimage import filters
from skimage import color


Waggle = namedtuple('Waggle', ['xs', 'ys', 'ts'])
# TODO add timestamp and maybe confidence

class FrequencyDetector:
    def __init__(self, buffer_size=32, width=160, height=120, fps=100,
                 freq_min=12, freq_max=14, num_base_functions=10, num_shifts=2):
        self.buffer_size = buffer_size
        self.width = width
        self.height = height 
        self.fps = fps
        self.buffer_time = (1 / fps) * buffer_size
        
        self.responses = np.zeros((buffer_size, num_base_functions, num_shifts, height, width), dtype=np.float32)
        self.activity = np.zeros((num_base_functions, num_shifts, height, width), dtype=np.float32)
        
        self.buffer = np.zeros((buffer_size, height, width), dtype=np.float32)
        self.buffer_idx = 0
        
        self.frequencies = np.linspace(freq_min, freq_max + 1, num=num_base_functions)
        self.functions = [self._get_base_function(f, num_shifts) for f in self.frequencies]
        self.functions = np.stack(self.functions, axis=0)[:, :, :, None, None]
        
    def _get_base_function(self, frequency, num_shifts=2):
        values = (np.linspace(0, (self.buffer_size / self.fps) * 2 * np.pi, num=self.buffer_size) * frequency)
        sin_values = [np.sin(values + factor * np.pi * 2 * np.pi) for factor in range(num_shifts)]
        return np.stack(sin_values).astype(np.float32)
    
    @profile
    def process(self, frame):
        self.buffer[self.buffer_idx] = frame
        frame_diff = self.buffer[None, None, self.buffer_idx, :, :] - \
            self.buffer[None, None, (self.buffer_idx-1) % self.buffer_size, :, :]
        #frame_diff /= np.abs(self.buffer[None, None, self.buffer_idx, :, :])
        
        self.activity -= self.responses[self.buffer_idx]
        
        self.responses[self.buffer_idx] = self.functions[:, :, self.buffer_idx, :, :] * frame_diff
            
        self.activity += self.responses[self.buffer_idx]
            
        self.buffer_idx += 1
        self.buffer_idx %= self.buffer_size
        
        return (self.activity ** 2).sum(axis=(0, 1))


class WaggleDetector:
    def __init__(self, max_distance, binarization_threshold,
                 max_frame_distance, min_num_detections,
                 dilation_selem_radius=6//2):
        self.max_distance = max_distance
        self.binarization_threshold = binarization_threshold
        self.max_frame_distance = max_frame_distance
        self.min_num_detections = min_num_detections
        self.selem = morphology.selem.disk(dilation_selem_radius)
        
        self.current_waggles = []
        self.finalized_waggles = []
        
    @profile
    def finalize_frames(self, frame_idx):
        new_current_waggles = []
        
        for waggle_idx, waggle in enumerate(self.current_waggles):
            if (frame_idx - waggle.ts[-1]) > self.max_frame_distance:
                if (len(waggle.ts)) > self.min_num_detections:
                    self.finalized_waggles.append(waggle)
                    # store waggle
                    pass
                else:
                    # discard waggle
                    pass
            else:
                new_current_waggles.append(waggle)
        self.current_waggles = new_current_waggles
        
    @profile
    def _get_activity_regions(self, activity):
        blobs = activity > self.binarization_threshold
        blobs_morph = morphology.opening(blobs)
        blobs_morph = morphology.dilation(blobs_morph, self.selem)
        blobs_labels = measure.label(blobs_morph, background=0)
        
        frame_waggle_positions = []
        for blob_index in range(1, blobs_labels.max() + 1):
            y, x = np.mean(np.argwhere(blobs_labels == blob_index), axis=0)
            frame_waggle_positions.append((y, x))
            
        return frame_waggle_positions
    
    @profile
    def _assign_regions_to_waggles(self, frame_idx, frame_waggle_positions):
        if len(self.current_waggles) == 0 and len(frame_waggle_positions) > 0:
            for region_idx in range(len(frame_waggle_positions)):
                self.current_waggles.append(
                    Waggle(xs=[frame_waggle_positions[region_idx][1]],
                           ys=[frame_waggle_positions[region_idx][0]],
                           ts=[frame_idx]))
            return
        
        if len(frame_waggle_positions) > 0:
            current_waggle_positions = [(w.ys[-1], w.xs[-1]) for w in self.current_waggles]
            current_waggle_frame_indices = [w.ts[-1] for w in self.current_waggles]

            # spatial and temporal distances of all activity regions to all
            # waggles currently being tracked
            assign_costs = scipy.spatial.distance.cdist(current_waggle_positions, frame_waggle_positions)
            time_dists = scipy.spatial.distance.cdist([[i] for i in current_waggle_frame_indices], [[frame_idx]])

            # never assign to waggles with distance higher than threshold
            # https://github.com/scipy/scipy/issues/6900
            assign_costs[assign_costs > self.max_distance] = 1e8
            for waggle_index, time_dist in enumerate(time_dists):
                if time_dist > self.max_frame_distance:
                    # never assign to an old waggle
                    assign_costs[waggle_index, :] = 1e8
            
            # assign activity regions to currently tracked waggles using hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(assign_costs)
            
            unassigned_fpws = set(range(len(frame_waggle_positions)))
            for waggle_idx, region_idx in zip(row_ind, col_ind):
                # hungarian algorithm can potentially assign all regions, even those
                # with a very high cost (1e8). make sure we never assign these
                assign = assign_costs[waggle_idx, region_idx] <= self.max_distance
                assign &= (frame_idx - current_waggle_frame_indices[waggle_idx]) <= self.max_frame_distance
                
                if assign:
                    self.current_waggles[waggle_idx].xs.append(frame_waggle_positions[region_idx][1])
                    self.current_waggles[waggle_idx].ys.append(frame_waggle_positions[region_idx][0])
                    self.current_waggles[waggle_idx].ts.append(frame_idx)
                
                    unassigned_fpws.discard(region_idx)
                
            # unassigned activity regions become new tracked waggles
            for region_index in unassigned_fpws:
                self.current_waggles.append(
                    Waggle(xs=[frame_waggle_positions[region_index][1]],
                           ys=[frame_waggle_positions[region_index][0]],
                           ts=[frame_idx]))
                
    @profile
    def process(self, frame_idx, activity):
        self.finalize_frames(frame_idx)
        
        frame_waggle_positions = self._get_activity_regions(activity)
        self._assign_regions_to_waggles(frame_idx, frame_waggle_positions)


class Camera:
    def __init__(self, height, width, fps, device):
        self.fps = fps
        self.height = height
        self.width = width
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        
    @profile
    def get_frame(self):
        ret, frame_orig = self.cap.read()
        
        if not ret:
            return ret, frame_orig, frame_orig
        
        frame = ((cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY) / 255) * 2) - 1

        if frame.shape != (self.height, self.width):
            frame = resize(frame, (self.height, self.width), mode='constant', anti_aliasing=True)

        return ret, frame, frame_orig
    
    def warmup(self, tolerance=0.01, num_frames_per_round=50, num_hits=2):
        fps_target = self.fps - tolerance * self.fps
        
        print('Camera warmup, FPS target >= {:.1f}'.format(fps_target))
        
        hits = 0
        while True:
            frame_idx = 0
            start_time = time.time()
            
            for _ in range(num_frames_per_round):
                ret, _, _ = self.get_frame()
                assert ret
                
                frame_idx += 1
                
            end_time = time.time()
            processing_fps = frame_idx / (end_time - start_time)
            fps_target_hit = processing_fps > fps_target
            
            if fps_target_hit:
                hits += 1
            else:
                hits = 0
                
            print('FPS: {:.1f} [{}]'.format(processing_fps, fps_target_hit))    
            
            if hits >= num_hits:
                print('Success')
                break

@click.command()
@click.option('--video_device', help='OpenCV video device. Can be camera index or video path')
@click.option('--height', default=120, help='Video frame height in px')
@click.option('--width', default=160, help='Video frame width in px')
@click.option('--fps', default=100, help='Frames per second')
@click.option('--bee_length', default=160, help='Approximate length of a bee in px')
@click.option('--binarization_threshold', default=3.5, help='Binarization threshold for waggle detection in log scale. Can be used to tune sensitivity/specitivity')
@click.option('--max_frame_distance', default=0.05, help='Maximum time inbetween frequency detections within one waggle in seconds')
@click.option('--min_num_detections', default=0.2, help='Minimum time of a waggle in seconds')
@click.option('--debug', default=False, help='Enable debug outputs/visualization')
def run(video_device, height, width, fps, bee_length, binarization_threshold, max_frame_distance, min_num_detections, debug):
    max_distance = bee_length / 4
    binarization_threshold = np.expm1(binarization_threshold)
    max_frame_distance = max_frame_distance * fps
    min_num_detections = min_num_detections * fps

    dd = FrequencyDetector(height=height, width=width, fps=fps)
    wd = WaggleDetector(max_distance=max_distance, binarization_threshold=binarization_threshold, 
                        max_frame_distance=max_frame_distance, min_num_detections=min_num_detections)
    cam = Camera(width=width, height=height, fps=fps, device=video_device)
    cam.warmup()

    frame_idx = 0
    start_time = time.time()
    while True:
        if frame_idx % 1000 == 0:
            start_time = time.time()

        ret, frame, frame_orig = cam.get_frame()
        
        if not ret:
            print('Unable to retrieve frame from video device')
            break
        
        activity = dd.process(frame)
        wd.process(frame_idx, activity)
        
        if debug and frame_idx % 11 == 0:
            current_waggle_num_detections = [len(w.xs) for w in wd.current_waggles]
            current_waggle_positions = [(w.ys[-1], w.xs[-1]) for w in wd.current_waggles]
            for blob_index, ((y, x), nd) in enumerate(zip(current_waggle_positions, current_waggle_num_detections)):
                cv2.circle(frame_orig, (int(x * 4), int(y * 4)), 10, (0, 0, 255), 5)
                    
            current_waggle_num_detections = [len(w.xs) for w in wd.finalized_waggles]
            current_waggle_positions = [(w.ys[-1], w.xs[-1]) for w in wd.finalized_waggles]
            for blob_index, ((y, x), nd) in enumerate(zip(current_waggle_positions, current_waggle_num_detections)):
                cv2.circle(frame_orig, (int(x * 4), int(y * 4)), 10, (255, 255, 255), 5)
                    
            cv2.imshow('Image', frame_orig)
            cv2.imshow('FrequencyDetector', activity)
            
            cv2.waitKey(1)
        
        end_time = time.time()
        processing_fps = ((frame_idx % 1000) + 1) / (end_time - start_time)
        frame_idx = (frame_idx + 1)
        
        sys.stdout.write('\rCurrently processing with {:.1f} fps '.format(processing_fps))
        
    cap.release()


if __name__ == '__main__':
    run()

