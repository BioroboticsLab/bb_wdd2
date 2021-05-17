import numpy as np
import cv2
from datetime import datetime
from imageio import imsave
import time
import os


try:
    import PyCapture2
except ImportError:
    print("Unable to import PyCapture2, Flea3 cameras won't work")


class Camera:
    def __init__(self, height, width, background=None, no_background_updates=False, alpha=None):
        self.background = background
        self.no_background_updates = no_background_updates
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = 0.95
        self.height = height
        self.width = width
        self.counter = 0

    def _get_frame(self):
        raise NotImplementedError()

    def get_frame(self):
        ret, frame_orig, timestamp = self._get_frame()

        if not ret:
            return ret, frame_orig, frame_orig, timestamp

        # FIXME: temporary hack to speed up image aquisition using Flea3 in the BeesBook setup
        # frame = resize(frame_orig, (self.height, self.width), mode='constant', order=1, anti_aliasing=False)
        frame = frame_orig[::2, ::2]

        if not self.no_background_updates:
            if self.background is None:
                self.background = np.copy(frame)
            else:
                if self.counter % 100 == 0:
                    self.background = self.alpha * self.background + (1 - self.alpha) * frame
        else:
            if self.background is None:
                raise RuntimeError("Background updates disabled but no existing background has been loaded.")

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
        self, height, width, fps, device, background=None, no_background_updates=False, alpha=None, fullframe_path=None
    ):
        super().__init__(height, width, background, alpha=alpha, no_background_updates=no_background_updates)

        self.fps = fps
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    def _get_frame(self):
        ret, frame_orig = self.cap.read()
        frame_orig = ((cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY) / 255) * 2) - 1
        return ret, frame_orig


class Flea3Capture(Camera):
    def __init__(
        self, height, width, fps, device, background=None, no_background_updates=False, alpha=None, fullframe_path=None, gain=100
    ):
        super().__init__(height, width, background, alpha=alpha, no_background_updates=no_background_updates)

        self.fps = fps
        self.device = device
        self.fullframe_path = fullframe_path
        self.counter = 0

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
        timestamp = datetime.utcnow().isoformat()

        # store on full frame image every hour
        if self.counter % (self.fps * 60 * 60) == 0:
            fullframe_im_path = os.path.join(
                self.fullframe_path, "{}-{}.png".format(self.device, datetime.utcnow())
            )
            print("\nStoring full frame image: {}".format(fullframe_im_path))
            imsave(fullframe_im_path, im)

        self.counter += 1

        im = (im[::3, ::3].astype(np.float32) / 255) * 2 - 1

        return True, im, timestamp


def cam_generator(cam_object, *args, **kwargs):
    cam = cam_object(*args, **kwargs)
    cam.warmup()

    while True:
        ret, frame, frame_orig, timestamp = cam.get_frame()
        if not ret:
            break
        yield ret, frame, frame_orig, cam.background, timestamp
