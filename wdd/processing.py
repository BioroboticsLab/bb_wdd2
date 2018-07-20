import cv2
from datetime import datetime
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from skimage import morphology
from skimage import measure
from skimage.morphology.selem import _default_selem

from wdd.datastructures import Waggle


class FrequencyDetector:
    def __init__(self, buffer_size=32, width=160, height=120, fps=100,
                 freq_min=11, freq_max=15, num_base_functions=5, num_shifts=2):
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
    
    def process(self, frame, background):
        self.buffer[self.buffer_idx] = frame
        frame_diff = self.buffer[None, None, self.buffer_idx, :, :] - \
            self.buffer[None, None, (self.buffer_idx-1) % self.buffer_size, :, :]
        frame_diff /= background[None, None, :, :] + 1
        
        self.activity -= self.responses[self.buffer_idx]
        
        self.responses[self.buffer_idx] = self.functions[:, :, self.buffer_idx, :, :] * frame_diff
            
        self.activity += self.responses[self.buffer_idx]
            
        self.buffer_idx += 1
        self.buffer_idx %= self.buffer_size
        
        return (self.activity ** 2).sum(axis=(0, 1)), frame_diff
    
    
class WaggleDetector:
    def __init__(self, max_distance, binarization_threshold,
                 max_frame_distance, min_num_detections,
                 exporter, dilation_selem_radius, debug=False):
        self.max_distance = max_distance
        self.binarization_threshold = binarization_threshold
        self.max_frame_distance = max_frame_distance
        self.min_num_detections = min_num_detections
        self.exporter = exporter
        self.default_selem = _default_selem(2).astype(np.uint8)
        self.selem = morphology.selem.disk(dilation_selem_radius)
        self.debug = debug
        
        self.current_waggles = []
        if debug:
            self.finalized_waggles = []
        
    def finalize_frames(self, frame_idx):
        new_current_waggles = []
        
        for waggle_idx, waggle in enumerate(self.current_waggles):
            if (frame_idx - waggle.ts[-1]) > self.max_frame_distance:
                if (len(waggle.ts)) > self.min_num_detections:
                    if self.debug:
                        self.finalized_waggles.append(waggle)
                    self.exporter.export(frame_idx, waggle)
                    pass
                else:
                    # discard waggle
                    pass
            else:
                new_current_waggles.append(waggle)
        self.current_waggles = new_current_waggles
        
    def _get_activity_regions(self, activity):
        blobs = (activity > self.binarization_threshold).astype(np.uint8)
        blobs_morph = cv2.morphologyEx(blobs, cv2.MORPH_OPEN, self.default_selem)
        blobs_morph = cv2.dilate(blobs_morph, self.selem)
        blobs_labels = measure.label(blobs_morph, background=0)
        
        frame_waggle_positions = []
        for blob_index in range(1, blobs_labels.max() + 1):
            y, x = np.mean(np.argwhere(blobs_labels == blob_index), axis=0)
            frame_waggle_positions.append((y, x))
            
        return frame_waggle_positions
    
    def _assign_regions_to_waggles(self, frame_idx, frame_waggle_positions):
        if len(self.current_waggles) == 0 and len(frame_waggle_positions) > 0:
            for region_idx in range(len(frame_waggle_positions)):
                self.current_waggles.append(
                    Waggle(timestamp=datetime.utcnow(),
                           xs=[frame_waggle_positions[region_idx][1]],
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
                    Waggle(timestamp=datetime.utcnow(),
                           xs=[frame_waggle_positions[region_index][1]],
                           ys=[frame_waggle_positions[region_index][0]],
                           ts=[frame_idx]))
                
    def process(self, frame_idx, activity, warmup=100):
        self.finalize_frames(frame_idx)
        
        frame_waggle_positions = self._get_activity_regions(activity)
        
        if frame_idx > warmup:
            self._assign_regions_to_waggles(frame_idx, frame_waggle_positions)
