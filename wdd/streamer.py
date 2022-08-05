import queue
import threading
import time
class RTMPStreamer:

    def __init__(self, address, input_fps=None, output_fps=None, debug=False):
        self.address = address
        
        output_params = output_params = {
            "-clones": ["-f", "lavfi", "-i", "anullsrc"],
            "-vcodec": "libx264",
            "-preset": "medium",
            "-b:v": "4500k",
            "-bufsize": "512k",
            "-pix_fmt": "yuv420p",
            "-f": "flv",
        }

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

        self.image_queue = queue.Queue(maxsize=3)

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

        while True:
            frame = self.image_queue.get()
            if frame is None:
                break
            self.stream.write(frame)