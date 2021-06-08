from datetime import datetime
import numpy as np
import os
from os.path import join
from os import makedirs
import json
import imageio
import queue
import threading

class WaggleSerializer:
    def __init__(
        self,
        cam_id,
        output_path
    ):

        self.cam_id = cam_id
        self.output_path = output_path

    def __call__(self, waggle, full_frame_rois, metadata_dict, **kwargs):
        dt = waggle.timestamp
        y, m, d, h, mn = dt.year, dt.month, dt.day, dt.hour, dt.minute
        waggle_path = join(
            self.output_path, str(self.cam_id), str(y), str(m), str(d), str(h), str(mn)
        )
        makedirs(waggle_path, exist_ok=True)
        waggle_idx = len(
            list(
                filter(
                    lambda x: os.path.isdir(os.path.join(waggle_path, x)), os.listdir(waggle_path)
                )
            )
        )
        waggle_path = join(waggle_path, str(waggle_idx))
        makedirs(waggle_path, exist_ok=True)

        print(
            "\r{} - {}: Saving new waggle: {}{}".format(self.cam_id, datetime.utcnow(), waggle_path, " " * 20)
        )

        for im_idx, roi in enumerate(full_frame_rois):
            roi = (roi * 255.0).astype(np.uint8)
            imageio.imwrite(join(waggle_path, "{:03d}.png".format(im_idx)), roi)

        with open(join(waggle_path, "waggle.json"), "w") as f:
            json.dump(metadata_dict, f)

        kwargs["output_path"] = waggle_path
        return waggle, full_frame_rois, metadata_dict, kwargs
        
class WaggleExportPipeline:
    def __init__(
        self,
        cam_id,
        full_frame_buffer,
        full_frame_buffer_len,
        full_frame_buffer_roi_size,
        datetime_buffer,
        min_images=32,
        subsampling_factor=None,
        export_steps=None,
        roi=None,
    ):

        self.cam_id = cam_id
        self.full_frame_buffer = full_frame_buffer
        self.full_frame_buffer_len = full_frame_buffer_len
        self.full_frame_buffer_roi_size = full_frame_buffer_roi_size
        self.datetime_buffer = datetime_buffer
        self.pad_size = self.full_frame_buffer_roi_size // 2
        self.min_images = min_images
        self.subsampling_factor = subsampling_factor
        self.roi = roi
        self.export_steps = export_steps

        self.export_queue = queue.Queue(maxsize=100)
        self.export_thread = threading.Thread(target=self.process_export_jobs, args=())
        self.export_thread.daemon = True
        self.export_thread.start()

    def finalize_exports(self):
        if not self.export_queue.empty():
            print("...please wait for exports to finish...", flush=True)
        self.export_queue.put(None)
        self.export_thread.join()

    def process_export_jobs(self):
        while True:
            waggle_data = self.export_queue.get()
            if waggle_data is None:
                return
            waggle, full_frame_rois, metadata_dict = waggle_data
            kwargs = dict()
            for step in self.export_steps:
                waggle_data = step(waggle, full_frame_rois, metadata_dict, **kwargs)
                if waggle_data is None:
                    break
                waggle, full_frame_rois, metadata_dict, kwargs = waggle_data

    def export(self, frame_idx, waggle):
        waggle_data = self.prepare_export(frame_idx, waggle)
        self.export_queue.put(waggle_data)

    def prepare_export(self, frame_idx, waggle):
        
        frame_idx_offset = frame_idx - waggle.ts[0] - 20
        if frame_idx_offset >= self.full_frame_buffer_len:
            frame_idx_offset = self.full_frame_buffer_len - 1
            print("Warning: Waggle ({}) longer than frame buffer size".format(waggle.timestamp))
        elif frame_idx_offset < self.min_images:
            frame_idx_offset = self.min_images

        # FIXME: scaling factor should depend on camera resolution
        center_x = int(np.median(waggle.xs)) * self.subsampling_factor
        center_y = int(np.median(waggle.ys)) * self.subsampling_factor
        if self.roi is not None:
            center_x += self.roi[0]
            center_y += self.roi[1]

        assert center_y < self.full_frame_buffer[0].shape[0]
        assert center_x < self.full_frame_buffer[0].shape[1]

        roi_x0, roi_x1 = center_x - self.pad_size, center_x + self.pad_size
        roi_y0, roi_y1 = center_y - self.pad_size, center_y + self.pad_size
        
        all_rois = []
        frame_timestamps = []
        for im_idx, idx in enumerate(range(frame_idx - frame_idx_offset, frame_idx)):
            idx %= self.full_frame_buffer_len
            roi = self.full_frame_buffer[idx]
            full_height, full_width = roi.shape[:2]
            roi = roi[
                max(0, roi_y0):min(full_height, roi_y1),
                max(0, roi_x0):min(full_width, roi_x1)
            ]
            if len(roi.shape) > 2:
                roi = np.mean(roi, axis=2)

            roi_target = np.mean(roi) * np.ones(shape=(self.full_frame_buffer_roi_size, self.full_frame_buffer_roi_size), dtype=np.float32)
            to_end_x = roi_target.shape[1] - (roi_x1 - min(full_width, roi_x1))
            to_end_y = roi_target.shape[0] - (roi_y1 - min(full_height, roi_y1))
            roi_target[
                (max(0, roi_y0) - roi_y0):to_end_y,
                (max(0, roi_x0) - roi_x0):to_end_x] = roi
            if roi.max() > 1.0 + 1e-3:
                roi_target /= 255.0

            frame_timestamps.append(self.datetime_buffer[idx])
            all_rois.append(roi_target)
            
        global_offset_x, global_offset_y = 0, 0
        if self.roi is not None:
            global_offset_x, global_offset_y = self.roi[:2]

        metadata = {
                "roi_coordinates": [[roi_x0 - self.pad_size, roi_y0 - self.pad_size], [roi_x1 - self.pad_size, roi_y1 - self.pad_size]],
                "roi_center": [center_x - self.pad_size, center_y - self.pad_size],
                "timestamp_begin": waggle.timestamp.isoformat(),
                "x_coordinates": [self.subsampling_factor * x + global_offset_x for x in waggle.xs],
                "y_coordinates": [self.subsampling_factor * y + global_offset_y for y in waggle.ys],
                "responses": [float(r) for r in waggle.responses],
                "frame_timestamps": [ts.isoformat() for ts in frame_timestamps],
                "camera_timestamps": [ts.isoformat() for ts in waggle.camera_timestamps],
                "frame_buffer_indices": [ts % self.full_frame_buffer_len for ts in waggle.ts],
                "subsampling": self.subsampling_factor,
                "global_roi": self.roi,
                "cam_id": self.cam_id,
            }

        return waggle, all_rois, metadata

