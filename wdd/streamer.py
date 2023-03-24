import datetime
import queue
import threading
import time

class RTMPStreamer:

    def __init__(self, address, input_fps=None, output_fps=None, debug=False, stream_codec="libx264",
                max_queue_size=3, force_format="flv", **kwargs):

        if "{datetime}" in address:
            dt_string = datetime.datetime.utcnow().isoformat().replace(":", "_")
            address = address.replace("{datetime}", dt_string)

        self.address = address
        
        output_params = output_params = {
            "-clones": ["-f", "lavfi", "-i", "anullsrc"],
            "-vcodec": stream_codec,
            "-preset": "medium",
            "-b:v": "4500k",
            "-bufsize": "512k",
            "-pix_fmt": "yuv420p",
            "-c:a": "aac",
            "-r": output_fps if output_fps else input_fps,
        }
        if force_format:
            output_params["-f"] = force_format

        output_params = {**output_params, **kwargs}

        from vidgear.gears import WriteGear

        self.stream = WriteGear(
            output_filename=self.address,
            logging=debug,
            **output_params
        )

        self.stream_every_x_frame = None
        self.frame_count = 0

        if input_fps is not None and output_fps is not None and (input_fps > output_fps):
            self.stream_every_x_frame = int(input_fps / output_fps)

        self.image_queue = queue.Queue(maxsize=max_queue_size)

        self.thread = threading.Thread(target=self.process_queue, args=())
        self.thread.daemon = False
        self.thread.start()

    def write(self, frame):

        if not self.stream_every_x_frame or (self.frame_count % self.stream_every_x_frame) == 0:
            try:
                self.image_queue.put_nowait(frame)
            except queue.Full:
                pass

        self.frame_count += 1

    def close(self):

        # Remove all from queue.
        try:
            while True:
                self.image_queue.get_nowait()
        except Exception as e:
            pass
        # Add "close now" marker to queue.
        self.image_queue.put(None)
        self.thread.join()

        if self.stream is not None:
            self.stream.close()

    def process_queue(self):

        error_count = 0
        while True:
            frame = self.image_queue.get()
            if frame is None:
                break

            try:
                self.stream.write(frame)
            except Exception as e:
                print("Error writing to stream. Retrying. {}".format(str(e)))
                error_count += 1
                time.sleep(5.0)
                if error_count >= 3:
                    break

