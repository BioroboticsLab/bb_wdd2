from datetime import datetime
import numpy as np
import os
from os.path import join
from os import makedirs
import json
import imageio
import queue
import threading
import hashlib
import uuid

def generate_64bit_id():
    """Returns a unique ID that is 64 bits long.
    Taken from the bb_pipeline codebase.
    """

    hasher = hashlib.sha1()
    hasher.update(uuid.uuid4().bytes)
    hash = int.from_bytes(hasher.digest(), byteorder='big')
    # strip to 64 bits
    hash = hash >> (hash.bit_length() - 64)
    return hash

class ClassFilter:
    def __init__(self, include_classes=[]):

        self.include_classes = set(include_classes)

    def __call__(self, waggle, full_frame_rois, metadata_dict, **kwargs):

        label = metadata_dict.get("predicted_class_label", None)

        if label:
            if self.include_classes and (label not in self.include_classes):
                return None
        
        return waggle, full_frame_rois, metadata_dict, kwargs


class VideoWriter:
    def __init__(self, device_name, fps, codec, directory="./", prefix="WDD_Recording_"):
        self.writer = None
        self.device_name = device_name
        self.codec = codec
        self.fps = fps
        self.is_color = None
        self.directory = directory
        self.prefix = prefix

    def write(self, image):
        if image is None:
            return

        if self.writer is None:
            import cv2
            if len(self.codec) != 4:
                fourcc = -1
            else:
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.is_color = len(image.shape) > 2 and image.shape[2] > 1
            current_datetime = datetime.utcnow().isoformat().replace(":", "_")
            mangled_device_name = self.device_name
            if "/" in mangled_device_name or "\\" in mangled_device_name:
                mangled_device_name = mangled_device_name.replace("\\", "/")
                mangled_device_name = mangled_device_name.split("/")[-1]
            output_video_name = self.directory + "/" + self.prefix + mangled_device_name + "_" + current_datetime + "+00.avi"
            print("Creating output video {}.".format(output_video_name))
            self.writer = cv2.VideoWriter(output_video_name,
                            fourcc,
                            int(self.fps),
                            (image.shape[1], image.shape[0]),
                            self.is_color)
        
        if not self.is_color and len(image.shape) > 2:
            image = image[:, :, 0]
        self.writer.write(image)

    def close(self):
        if self.writer is not None:
            self.writer.release()


class WaggleSerializer:
    def __init__(
        self,
        cam_id,
        output_path
    ):

        self.cam_id = cam_id
        self.output_path = output_path

        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self.serialize, args=())
        self.thread.daemon = False
        self.thread.start()

    def __call__(self, waggle, full_frame_rois, metadata_dict, **kwargs):

        self.queue.put((waggle, full_frame_rois, metadata_dict, kwargs))

        return waggle, full_frame_rois, metadata_dict, kwargs

    def serialize(self):

        while True:

            data = self.queue.get()
            if data is None:
                break

            waggle, full_frame_rois, metadata_dict, kwargs = data

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

            label = metadata_dict.get("predicted_class_label", "waggle")
            print(
                "\r{} - {}: Saving new {}: {}{}".format(self.cam_id, datetime.utcnow(), label, waggle_path, " " * 20)
            )

            for im_idx, roi in enumerate(full_frame_rois):
                roi = (roi * 255.0).astype(np.uint8)
                imageio.imwrite(join(waggle_path, "{:03d}.png".format(im_idx)), roi)

            with open(join(waggle_path, "waggle.json"), "w") as f:
                json.dump(metadata_dict, f)

    def finalize_serialization(self):
        if self.queue is not None:
            self.queue.put(None)
        self.thread.join()
        
class WaggleExportPipeline:
    def __init__(
        self,
        cam_id,
        full_frame_buffer,
        full_frame_buffer_len,
        full_frame_buffer_roi_size,
        datetime_buffer,
        fps=None,
        min_images=None,
        subsampling_factor=None,
        export_steps=None,
        roi=None,
    ):

        self.cam_id = cam_id
        self.full_frame_buffer = full_frame_buffer
        self.full_frame_buffer_len = full_frame_buffer_len
        self.full_frame_buffer_roi_size = (full_frame_buffer_roi_size // 2) * 2 # Force crop being even.
        self.datetime_buffer = datetime_buffer
        self.pad_size = self.full_frame_buffer_roi_size // 2
        self.fps = fps
        self.min_images = min_images
        self.subsampling_factor = subsampling_factor
        self.roi = roi
        self.export_steps = export_steps

        self.export_queue = queue.Queue(maxsize=100)
        self.export_thread = threading.Thread(target=self.process_export_jobs, args=())
        self.export_thread.daemon = True
        self.export_thread.start()

    def finalize_exports(self):
        if self.export_queue is not None:
            if not self.export_queue.empty():
                print("...please wait for exports to finish...", flush=True)
            self.export_queue.put(None)
        self.export_thread.join()

    def process_export_jobs(self):
        while True:
            waggle_data = self.export_queue.get()
            if waggle_data is None:
                return
            try:
                waggle, full_frame_rois, metadata_dict = waggle_data
                kwargs = dict()
                for step in self.export_steps:
                    waggle_data = step(waggle, full_frame_rois, metadata_dict, **kwargs)
                    if waggle_data is None:
                        break
                    waggle, full_frame_rois, metadata_dict, kwargs = waggle_data
            except Exception as e:
                # Catch the error, because this thread crashing will not stop the program anyway.
                print("\nError in serialization thread:")
                import traceback
                traceback.print_exc()
                print(str(e), flush=True)
                # This, however, will lead to an error later.
                self.export_queue = None

                exit(1)
                return

    def export(self, frame_idx, waggle):
        if self.export_queue is None:
            raise RuntimeError("Exporting thread not running.")
        waggle_data = self.prepare_export(frame_idx, waggle)
        self.export_queue.put(waggle_data)

    def prepare_export(self, frame_idx, waggle):
        
        # Store a second of images before waggle start.
        frame_idx_offset = frame_idx - waggle.ts[0] + int(self.fps)
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
                "waggle_id": generate_64bit_id()
            }

        return waggle, all_rois, metadata

