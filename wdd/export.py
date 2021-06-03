from datetime import datetime
import numpy as np
import os
from os.path import join
from os import makedirs
import json
import imageio


class WaggleExporter:
    def __init__(
        self,
        cam_id,
        output_path,
        full_frame_buffer,
        full_frame_buffer_len,
        full_frame_buffer_roi_size,
        datetime_buffer,
        min_images=32,
        subsampling_factor=None,
        save_data_fn=None
    ):
        self.cam_id = cam_id
        self.output_path = output_path
        self.full_frame_buffer = full_frame_buffer
        self.full_frame_buffer_len = full_frame_buffer_len
        self.full_frame_buffer_roi_size = full_frame_buffer_roi_size
        self.datetime_buffer = datetime_buffer
        self.pad_size = self.full_frame_buffer_roi_size // 2
        self.min_images = min_images
        self.subsampling_factor = subsampling_factor
        # Allow overwriting the data export function (e.g. to calculate scores instead of saving).
        self.save_data_fn = save_data_fn
        if self.save_data_fn is None:
            self.save_data_fn = self._save_data

    def _save_data(self, waggle, full_frame_rois, metadata_dict):
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
            imageio.imwrite(join(waggle_path, "{:03d}.png".format(im_idx)), roi)

        with open(join(waggle_path, "waggle.json"), "w") as f:
            json.dump(metadata_dict, f)

    def export(self, frame_idx, waggle):
        
        frame_idx_offset = frame_idx - waggle.ts[0] - 20
        if frame_idx_offset >= self.full_frame_buffer_len:
            frame_idx_offset = self.full_frame_buffer_len - 1
            print("Warning: Waggle ({}) longer than frame buffer size".format(waggle.timestamp))
        elif frame_idx_offset < self.min_images:
            frame_idx_offset = self.min_images

        # FIXME: scaling factor should depend on camera resolution
        center_x = int(np.median(waggle.xs)) + self.pad_size
        center_y = int(np.median(waggle.ys)) + self.pad_size

        assert center_y < self.full_frame_buffer.shape[1]
        assert center_x < self.full_frame_buffer.shape[2]

        roi_x0, roi_x1 = max(0, center_x - self.pad_size), center_x + self.pad_size
        roi_y0, roi_y1 = max(0, center_y - self.pad_size), center_y + self.pad_size
        
        all_rois = []
        frame_timestamps = []
        for im_idx, idx in enumerate(range(frame_idx - frame_idx_offset, frame_idx)):
            idx %= self.full_frame_buffer_len
            roi = self.full_frame_buffer[
                idx,
                roi_y0:roi_y1,
                roi_x0:roi_x1,
            ]
            frame_timestamps.append(self.datetime_buffer[idx])
            all_rois.append((roi * 255.0).astype(np.uint8))
            

        metadata = {
                "roi_coordinates": [[roi_x0 - self.pad_size, roi_y0 - self.pad_size], [roi_x1 - self.pad_size, roi_y1 - self.pad_size]],
                "roi_center": [center_x - self.pad_size, center_y - self.pad_size],
                "timestamp_begin": waggle.timestamp.isoformat(),
                "x_coordinates": waggle.xs,
                "y_coordinates": waggle.ys,
                "responses": [float(r) for r in waggle.responses],
                "frame_timestamps": [ts.isoformat() for ts in frame_timestamps],
                "camera_timestamps": [ts.isoformat() for ts in waggle.camera_timestamps],
                "frame_buffer_indices": [ts % self.full_frame_buffer_len for ts in waggle.ts],
                "subsampling": self.subsampling_factor,
            }

        self.save_data_fn(waggle, all_rois, metadata)

