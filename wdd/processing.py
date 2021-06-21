import cv2
from datetime import datetime
import numpy as np
import opt_einsum
import scipy
import scipy.signal
from scipy.optimize import linear_sum_assignment
from skimage import morphology
from skimage import measure
from skimage.morphology.selem import _default_selem

from wdd.datastructures import Waggle

import numba

@numba.njit(parallel=True, cache=True)
def normalize_rolling_buffer_online(rolling_mean, buffer, output_buffer):
    """
    Takes a current mean and an image buffer (n_images, height, width).
    Updates the mean and writes a normalized image buffer to the output argument.
    """
    buffer_size = np.float32(buffer.shape[0])
    oldest_frame = buffer[0]
    newest_frame = buffer[-1]

    # Update rolling mean by subtracting first observation and adding latest.
    rolling_mean = rolling_mean - oldest_frame / buffer_size + newest_frame / buffer_size

    # Normalize buffer by subtracting mean and dividing by max(abs()).
    for y in numba.prange(buffer.shape[1]):
        for x in numba.prange(buffer.shape[2]):
            current_buffer_mean = rolling_mean[y, x]
            pixel_max = 1e-1
            for z in range(buffer.shape[0]):
                new_value = buffer[z, y, x] - current_buffer_mean
                new_value = np.abs(new_value)
                if new_value > pixel_max:
                    pixel_max = new_value
            for z in range(buffer.shape[0]):
                output_buffer[z, y, x] = (buffer[z, y, x] - current_buffer_mean) / pixel_max

@numba.njit(parallel=True, cache=True)
def apply_wavelets(wavelets, normalized_frame_buffer, output):
    """
    Multiply wavelets onto normalized frame buffer and sum over the absolute responses.
    """
    n_wavelets = np.float32(wavelets.shape[0])

    for y in numba.prange(normalized_frame_buffer.shape[1]):
        for x in numba.prange(normalized_frame_buffer.shape[2]):
            pixel_sum = 0.0
            for w in range(wavelets.shape[0]):
                response = 0.0
                for z in range(normalized_frame_buffer.shape[0]):
                    response += wavelets[w, z] * normalized_frame_buffer[z, y, x]
                if response < 0.0:
                    response = -response
                pixel_sum += response

            output[y, x] = pixel_sum / n_wavelets

@numba.njit(parallel=True, cache=True)
def post_process_response(frame_response, fps, activity, activity_long, activity_decay, activity_long_decay, output):
    """
    Post process responses by updating high-pass filter.
    Writes current response into output argument, also updates the two activity buffers.
    """
    for y in numba.prange(frame_response.shape[0]):
        for x in numba.prange(frame_response.shape[1]):
            v = frame_response[y, x] ** 2.0
            new_activity_value = activity_decay * activity[y, x] + v
            activity[y, x] = new_activity_value
            current_pixel_activity = new_activity_value - (activity_long[y, x] / fps)
            activity_long[y, x] = activity_long_decay * activity_long[y, x] + v
            
            if current_pixel_activity < 0.0:
                current_pixel_activity = 0.0
            else:
                current_pixel_activity = current_pixel_activity ** 4.0
            output[y, x] = current_pixel_activity

class FrequencyDetector:
    def __init__(
        self,
        width=160,
        height=120,
        fps=100,
        freq_min=12,
        freq_max=15,
        num_base_functions=4,
    ):
        self.width = width
        self.height = height
        self.fps = np.float32(fps)
        
        self.frequencies = np.linspace(freq_min, freq_max + 1, num=num_base_functions)
        self.wavelets = [self._get_wavelet(f) for f in self.frequencies]
        self.max_wavelet_length = max([w.shape[0] for w in self.wavelets])
        self.wavelets = np.stack([np.pad(w, (self.max_wavelet_length - w.shape[0], 0)) for w in self.wavelets])

        self.buffer_size = int(2.5 * self.max_wavelet_length)
        self.buffer_time = (1 / fps) * self.buffer_size

        self.activity_long = np.zeros((height, width), dtype=np.float32)
        self.activity = np.zeros((height, width), dtype=np.float32)

        self.buffer = np.zeros((self.buffer_size, height, width), dtype=np.float32)
        self.buffer_idx = 0

        self.frame_response = np.zeros(shape=(height, width), dtype=np.float32)

        # Decay background activity slower than 'current' activity.
        # The activity of 1 s ago will still contribute 5% to the 'current' activity.
        self.activity_decay = np.exp(np.log(0.05) / self.fps)
        self.activity_long_decay = np.exp(np.log(0.1) / self.fps)

        self.einsum_expression = opt_einsum.contract_expression("ij,jkl->ikl", self.wavelets.shape, (self.wavelets.shape[1], height, width))

        self.warmup_period_over = False
        self.rolling_mean = np.zeros(shape=(height, width), dtype=np.float32)
        self.temp_full_buffer = np.zeros(shape=(self.wavelets.shape[1], height, width), dtype=np.float32)
        self.temp_one_frame_buffer = np.zeros(shape=(height, width), dtype=np.float32)

    def _get_wavelet(self, frequency):
        s, w = 0.25, 15.0
        M = int(2 * s * w * self.fps / frequency)
        wavelet = scipy.signal.morlet(M, w, s, complete=False)
        wavelet = wavelet[:wavelet.shape[0]//2]
        wavelet = np.real(wavelet).astype(np.float32)
        wavelet /= np.abs(wavelet).max()

        return wavelet

    def process(self, frame):
        # Calculate rolling buffer index, so that the wavelets can always be multiplied
        # against a continuous buffer region.
        # I.e. sometimes write images at end AND beginning of buffer.
        current_buffer_idx = self.max_wavelet_length + self.buffer_idx - 1
        self.buffer[current_buffer_idx] = frame
        if (self.buffer_idx > (self.buffer_size - 2.0 * self.max_wavelet_length)):
            idx = self.max_wavelet_length - (self.buffer_size - current_buffer_idx)
            self.buffer[idx] = frame
        
        current_buffer = self.buffer[(current_buffer_idx - self.max_wavelet_length + 1):(current_buffer_idx + 1)]
        if not self.warmup_period_over:
            if self.buffer_idx >= self.max_wavelet_length:
                self.warmup_period_over = True
                # Initialize rolling sum.
                self.rolling_mean = np.mean(current_buffer, axis=0)
        else:
            normalize_rolling_buffer_online(self.rolling_mean, current_buffer, self.temp_full_buffer)
            apply_wavelets(self.wavelets, self.temp_full_buffer, self.frame_response)
            post_process_response(self.frame_response, self.fps, self.activity, self.activity_long, self.activity_decay, self.activity_long_decay, self.temp_one_frame_buffer)

        self.buffer_idx += 1
        self.buffer_idx %= (self.buffer_size - self.max_wavelet_length)

        return self.temp_one_frame_buffer


class WaggleDetector:
    def __init__(
        self,
        max_distance,
        binarization_threshold,
        max_frame_distance,
        min_num_detections,
        exporter,
        opening_selem_radius,
        dilation_selem_radius,
        datetime_buffer,
        full_frame_buffer_len,
        debug=False,
    ):
        self.max_distance = max_distance
        self.binarization_threshold = binarization_threshold
        self.max_frame_distance = max_frame_distance
        self.min_num_detections = min_num_detections
        self.exporter = exporter
        self.default_selem = _default_selem(opening_selem_radius).astype(np.uint8)
        self.selem = morphology.selem.disk(dilation_selem_radius)
        self.datetime_buffer = datetime_buffer
        self.full_frame_buffer_len = full_frame_buffer_len
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
        blobs_morph = (activity > self.binarization_threshold).astype(np.uint8)
        blobs_morph = cv2.morphologyEx(blobs_morph, cv2.MORPH_OPEN, self.default_selem)
        blobs_morph = cv2.dilate(blobs_morph, self.selem)
        blobs_labels = measure.label(blobs_morph, background=0)

        frame_waggle_positions = []
        for blob_index in range(1, blobs_labels.max() + 1):
            waggle_area = blobs_labels == blob_index
            y, x = np.mean(np.argwhere(waggle_area), axis=0)

            waggle_response = np.max(activity[waggle_area])
            frame_waggle_positions.append((y, x, waggle_response))

        return frame_waggle_positions

    def _assign_regions_to_waggles(self, frame_idx, frame_waggle_positions):
        if len(self.current_waggles) == 0 and len(frame_waggle_positions) > 0:
            for region_idx in range(len(frame_waggle_positions)):
                waggle_info = frame_waggle_positions[region_idx]
                waggle_timestamp = self.datetime_buffer[frame_idx % self.full_frame_buffer_len]
                self.current_waggles.append(
                    Waggle(
                        timestamp=waggle_timestamp,
                        xs=[waggle_info[1]],
                        ys=[waggle_info[0]],
                        ts=[frame_idx],
                        camera_timestamps=[waggle_timestamp],
                        responses=[waggle_info[2]],
                        system_timestamp=datetime.utcnow()
                    )
                )
            return

        if len(frame_waggle_positions) > 0:
            current_waggle_positions = [(w.ys[-1], w.xs[-1]) for w in self.current_waggles]
            current_waggle_frame_indices = [w.ts[-1] for w in self.current_waggles]

            # spatial and temporal distances of all activity regions to all
            # waggles currently being tracked
            assign_costs = scipy.spatial.distance.cdist(
                current_waggle_positions, [l[:2] for l in frame_waggle_positions]
            )
            time_dists = scipy.spatial.distance.cdist(
                [[i] for i in current_waggle_frame_indices], [[frame_idx]]
            )

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
                assign &= (
                    frame_idx - current_waggle_frame_indices[waggle_idx]
                ) <= self.max_frame_distance

                if assign:
                    waggle_info = frame_waggle_positions[region_idx]
                    existing_waggle = self.current_waggles[waggle_idx]
                    existing_waggle.xs.append(waggle_info[1])
                    existing_waggle.ys.append(waggle_info[0])
                    existing_waggle.responses.append(waggle_info[2])
                    existing_waggle.ts.append(frame_idx)
                    existing_waggle.camera_timestamps.append(
                        self.datetime_buffer[frame_idx % self.full_frame_buffer_len]
                    )

                    unassigned_fpws.discard(region_idx)

            # unassigned activity regions become new tracked waggles
            for region_index in unassigned_fpws:
                waggle_timestamp = self.datetime_buffer[frame_idx % self.full_frame_buffer_len]
                waggle_info = frame_waggle_positions[region_index]
                self.current_waggles.append(
                    Waggle(
                        timestamp=waggle_timestamp,
                        xs=[waggle_info[1]],
                        ys=[waggle_info[0]],
                        ts=[frame_idx],
                        camera_timestamps=[waggle_timestamp],
                        responses=[waggle_info[2]],
                        system_timestamp=datetime.utcnow()
                    )
                )

    def process(self, frame_idx, activity, warmup=100):
        self.finalize_frames(frame_idx)

        frame_waggle_positions = self._get_activity_regions(activity)

        if frame_idx > warmup:
            self._assign_regions_to_waggles(frame_idx, frame_waggle_positions)
