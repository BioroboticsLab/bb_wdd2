
class RTMPStreamer:

    def __init__(self, address, debug=False):
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


    def write(self, frame):
        self.stream.write(frame)

    def close(self):
        if self.stream is not None:
            self.stream.close()