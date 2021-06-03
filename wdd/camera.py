import numpy as np
import cv2
import datetime
import pytz
import imageio
import skimage.transform
import time
import os


try:
    import PyCapture2
except ImportError:
    print("Unable to import PyCapture2, Flea3 cameras won't work")


class Camera:
    def __init__(self, height, width, fps=None, subsample=0, fullframe_path=None, cam_identifier="cam", start_timestamp=None):
        self.fps = fps
        self.height = height
        self.width = width
        self.subsample = subsample
        self.counter = 0
        self.fullframe_path = fullframe_path
        self.cam_identifier = cam_identifier

        self.start_timestamp = None
        if start_timestamp is not None:
            try:
                self.start_timestamp = datetime.datetime.fromisoformat(start_timestamp).astimezone(pytz.utc)
            except Exception as e:
                raise ValueError("Invalid 'start_timestamp' argument: {}. Has to be iso-formatted.".format(str(e)))
        
        self.resize_warning_emitted = False
    def _get_frame(self):
        """
        Function should return a float32 image in the range of [0, 1].
        """
        raise NotImplementedError()
    
    def subsample_frame(self, frame):
        if (frame is not None) and (self.subsample > 1):
            frame = frame[::self.subsample, ::self.subsample]
        return frame

    def get_current_timestamp(self):
        if self.start_timestamp is not None:
            timestamp = self.start_timestamp + datetime.timedelta(seconds=(self.counter / self.fps))
        else:
            timestamp = datetime.datetime.utcnow()
        return timestamp

    def get_frame(self):
        ret, frame_orig, timestamp = self._get_frame()

        if not ret:
            return ret, frame_orig, frame_orig, timestamp

        if frame_orig.shape[0] != self.height or frame_orig.shape[1] != self.width:
            frame = skimage.transform.resize(frame_orig, (self.height, self.width), mode='constant', order=1, anti_aliasing=False)

            if not self.resize_warning_emitted:
                self.resize_warning_emitted = True
                print("Warning! Necessary to resize the image after subsampling ({}) to fit into desired output shape ({}). This could be slow.".format(frame_orig.shape, frame.shape))
        else:
            frame = frame_orig
            
        # store on full frame image every hour
        if self.fullframe_path and (self.counter % (self.fps * 60 * 60) == 0):
            fullframe_im_path = os.path.join(
                self.fullframe_path, "{}-{}.png".format(self.cam_identifier, timestamp.isoformat())
            )
            print("\nStoring full frame image: {}".format(fullframe_im_path))
            imageio.imwrite(fullframe_im_path, (frame_orig * 255.0).astype(np.uint8))

        self.counter += 1

        return ret, frame, frame_orig, timestamp

    def warmup(self, tolerance=0.01, num_frames_per_round=100, num_hits=3):
        fps_target = self.fps - tolerance * self.fps

        print("Camera warmup, FPS target >= {:.1f}".format(fps_target))

        hits = 0
        while True:
            frame_idx = 0
            start_time = time.time()

            for _ in range(num_frames_per_round):
                ret, _, _, _ = self.get_frame()
                assert ret

                frame_idx += 1

            end_time = time.time()
            processing_fps = frame_idx / (end_time - start_time)
            fps_target_hit = processing_fps > fps_target

            if fps_target_hit:
                hits += 1
            else:
                hits = 0

            print("FPS: {:.1f} [{}]".format(processing_fps, fps_target_hit))

            if hits >= num_hits:
                print("Success")
                break


class OpenCVCapture(Camera):
    def __init__(
        self, height, width, fps, device, subsample=0, fullframe_path=None, cam_identifier=None, start_timestamp=None
    ):
        super().__init__(height, width, fps=fps, subsample=subsample, fullframe_path=fullframe_path, cam_identifier=cam_identifier, start_timestamp=start_timestamp)

        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            try:
                # Maybe the user intended the argument as a device index.
                self.cap = cv2.VideoCapture(int(device))
            except:
                pass
            if not self.cap.isOpened():
                raise RuntimeError("Could not open OpenCV device '{}'!".format(device))
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

        # Reuse the same allocated memory for a small speedup.
        self.buffer_frame_rgb_float = None
        self.capture_double_buffer = None

    def _get_frame(self):
        ret, self.capture_double_buffer = self.cap.read(self.capture_double_buffer)
        timestamp = self.get_current_timestamp()

        frame_orig = self.subsample_frame(self.capture_double_buffer)
        if frame_orig is not None:
            if self.buffer_frame_rgb_float is None:
                self.buffer_frame_rgb_float = np.zeros(shape=frame_orig.shape, dtype=np.float32)
            self.buffer_frame_rgb_float[:] = frame_orig
            self.buffer_frame_rgb_float /= 255.0
            frame_orig = self.buffer_frame_rgb_float.mean(axis=2)
        return ret, frame_orig, timestamp


class Flea3Capture(Camera):
    def __init__(
        self, height, width, device, subsample=0,  fullframe_path=None, gain=100, cam_identifier=None, start_timestamp=None
    ):
        super().__init__(height, width, fps=fps, subsample=subsample, fullframe_path=fullframe_path, cam_identifier=cam_identifier, start_timestamp=start_timestamp)

        self.device = device

        bus = PyCapture2.BusManager()
        numCams = bus.getNumOfCameras()

        print(10 * "*" + " AVAILABLE CAMERAS " + 10 * "*")
        for i in range(numCams):
            print("\n{})".format(i + 1))
            self.cap = PyCapture2.Camera()
            self.uid = bus.getCameraFromIndex(i)
            self.cap.connect(self.uid)
            self.print_cam_info(self.cap)
            self.cap.disconnect()

            bus = PyCapture2.BusManager()

        if not numCams:
            raise RuntimeError("No Flea3 camera detected")

        self.cap = PyCapture2.Camera()
        self.uid = bus.getCameraFromSerialNumber(int(device))
        self.cap.connect(self.uid)

        print(10 * "*" + " SELECTED CAMERA " + 10 * "*")
        self.print_cam_info(self.cap)

        fmt7imgSet = PyCapture2.Format7ImageSettings(
            PyCapture2.MODE.MODE_4, 0, 0, 2048, 1080, PyCapture2.PIXEL_FORMAT.MONO8
        )
        fmt7pktInf, isValid = self.cap.validateFormat7Settings(fmt7imgSet)
        if not isValid:
            raise RuntimeError("Format7 settings are not valid!")

        self.cap.setFormat7ConfigurationPacket(fmt7pktInf.recommendedBytesPerPacket, fmt7imgSet)

        self.cap.setProperty(type=PyCapture2.PROPERTY_TYPE.FRAME_RATE, absValue=fps)
        self.cap.setProperty(type=PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE, absValue=False)
        self.cap.setProperty(type=PyCapture2.PROPERTY_TYPE.SHUTTER, absValue=1 / fps * 1000)
        self.cap.setProperty(type=PyCapture2.PROPERTY_TYPE.GAIN, absValue=100)

        self.cap.startCapture()

    def print_cam_info(self, cam):
        camInfo = cam.getCameraInfo()
        print()
        print("Serial number - ", camInfo.serialNumber)
        print("Camera model - ", camInfo.modelName)
        print("Camera vendor - ", camInfo.vendorName)
        print("Sensor - ", camInfo.sensorInfo)
        print("Resolution - ", camInfo.sensorResolution)
        print("Firmware version - ", camInfo.firmwareVersion)
        print("Firmware build time - ", camInfo.firmwareBuildTime)
        print()

    def _get_frame(self):
        # This requires a modified PyCapture2 with numpy bindings
        # https://github.com/GreenSlime96/PyCapture2_NumPy
        im = np.array(self.cap.retrieveBuffer())
        timestamp = self.get_current_timestamp()

        im = self.subsample_frame(im)
        im = im.astype(np.float32) / 255.0

        return True, im, timestamp


def cam_generator(cam_object, warmup=True, *args, **kwargs):
    cam = cam_object(*args, **kwargs)
    if warmup:
        cam.warmup()

    while True:
        ret, frame, frame_orig, timestamp = cam.get_frame()
        if not ret:
            break
        yield ret, frame, frame_orig, timestamp
