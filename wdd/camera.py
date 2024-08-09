import numpy as np
import cv2
import datetime
import pytz
import imageio
import skimage.transform
import time
import os

flea3_supported = False
try:
    import PySpin
    import simple_pyspin
    flea3_supported = True
except ImportError:
    print("Unable to import PySpin and simple_pyspin for Flea3 cameras. Trying legacy SDK..")

try:
    import PyCapture2
except ImportError:
    if not flea3_supported:
        print("Unable to import PyCapture2, Flea3 cameras won't work")

from .torch_support import torch


class Camera:
    def __init__(self, height, width, target_height, target_width, fps=None, subsample=0, fullframe_path=None, cam_identifier="cam", start_timestamp=None, stop_timestamp=None, roi=None, quiet=False):
        self.fps = fps
        self.height = height
        self.width = width
        self.target_height = target_height
        self.target_width = target_width
        self.subsample = subsample
        self.counter = 0
        self.fullframe_path = fullframe_path
        self.cam_identifier = cam_identifier
        if roi is not None:
            roi = list(map(int, [roi[0], roi[1], roi[0] + roi[2], roi[1] + roi[3]]))
        self.roi = roi

        self.start_timestamp = None
        if start_timestamp is not None:
            try:
                self.start_timestamp = datetime.datetime.fromisoformat(start_timestamp).astimezone(pytz.utc)
            except Exception as e:
                raise ValueError("Invalid 'start_timestamp' argument: {}. Has to be iso-formatted.".format(str(e)))
            
        self.stop_timestamp = None
        if stop_timestamp is not None:
            try:
                self.stop_timestamp = datetime.datetime.fromisoformat(stop_timestamp).astimezone(pytz.utc)
            except Exception as e:
                raise ValueError("Invalid 'stop_timestamp' argument: {}. Has to be iso-formatted.".format(str(e)))
            
        self.resize_warning_emitted = False
    def _get_frame(self):
        """
        Function should return a float32 image in the range of [0, 1].
        """
        raise NotImplementedError()
    
    def subsample_frame(self, frame, normalize=False):
        if frame is not None:
            if self.roi is not None:
                frame = frame[self.roi[1]:self.roi[3], self.roi[0]:self.roi[2]]
            if self.subsample > 1:
                frame = frame[::self.subsample, ::self.subsample]

            if torch is None and not np.issubdtype(frame.dtype, np.floating):
                frame = frame.astype(np.float32)
            elif torch is not None:
                if not frame.flags.writeable:
                    frame = frame.copy()
                frame = torch.as_tensor(frame, dtype=torch.float32, device="cuda")

            if len(frame.shape) > 2 and frame.shape[2] > 1:
                assert frame.shape[2] < frame.shape[0]
                frame = frame.mean(axis=2)

            if normalize:
                frame /= 255.0

        return frame

    def get_current_timestamp(self):
        if self.start_timestamp is not None:
            timestamp = self.start_timestamp + datetime.timedelta(seconds=(self.counter / self.fps))
        else:
            timestamp = pytz.UTC.localize(datetime.datetime.utcnow())
        return timestamp

    def get_frame(self):
        ret, frame, full_frame, timestamp = self._get_frame()

        if self.stop_timestamp is not None and timestamp is not None:
            if timestamp >= self.stop_timestamp:
                # Abort frame acquisition.
                ret = False

        if not ret:
            return ret, frame, full_frame, timestamp

        if frame.shape[0] != self.target_height or frame.shape[1] != self.target_width:
            frame_original_shape = frame.shape
            target_shape = (self.target_height, self.target_width)

            if not self.resize_warning_emitted:
                self.resize_warning_emitted = True
                print("Warning! Necessary to resize the image after subsampling ({}) to fit into desired output shape ({}). This could be slow.".format(frame_original_shape, target_shape))
            
            if torch is None:
                frame = skimage.transform.resize(frame, target_shape, mode='constant', order=1, anti_aliasing=False)
            else:
                from torchvision.transforms import Resize
                resizer = Resize(target_shape)
                frame = resizer(frame.unsqueeze(0)).squeeze()

            
        # store on full frame image every hour
        if self.fullframe_path and (self.counter % (self.fps * 60 * 60) == 0):
            fullframe_im_path = os.path.join(
                self.fullframe_path, "{}-{}.png".format(self.cam_identifier, timestamp.isoformat().replace(":", "_"))
            )
            print("\nStoring full frame image: {}".format(fullframe_im_path))
            output_frame = full_frame
            if np.max(output_frame) < 1.0 + 1e-3:
                output_frame = (output_frame * 255.0).astype(np.uint8)
            imageio.imwrite(fullframe_im_path, output_frame)

        self.counter += 1

        return ret, frame, full_frame, timestamp

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

            print("FPS: {:.1f} [{}]".format(processing_fps, fps_target_hit), flush=True)

            if hits >= num_hits:
                print("Success")
                break

    def stop(self):
        pass


class OpenCVCapture(Camera):
    def __init__(
        self, height, width, fps, device, subsample=0, fullframe_path=None, cam_identifier=None, start_timestamp=None, roi=None, fourcc=None, capture_api=None, quiet=False, **kwargs
    ):
        super().__init__(height, width, fps=fps, subsample=subsample, fullframe_path=fullframe_path, cam_identifier=cam_identifier, start_timestamp=start_timestamp, roi=roi, **kwargs)

        preferred_api = cv2.CAP_ANY
        if capture_api is not None:
            try:
                preferred_api = getattr(cv2, capture_api)
            except Exception as e:
                print("video_device_api invalid. Possible values are e.g. CAP_MSMF, CAP_DSHOW, CAP_FFMPEG, CAP_V4L2.")

        if not quiet: 
            print("Opening OpenCV video device...", end="", flush=True)
        self.cap = cv2.VideoCapture(device, preferred_api)
        if not self.cap.isOpened():
            try:
                # Maybe the user intended the argument as a device index.
                self.cap = cv2.VideoCapture(int(device), preferred_api)
            except:
                pass
            if not self.cap.isOpened():
                raise RuntimeError("Could not open OpenCV device '{}'!".format(device))
        
        if not quiet:
            print(" Done. Setting capture parameters...", end="", flush=True)

        if fourcc:
            if len(fourcc) != 4:
                raise ValueError("device_fourcc argument must be of length 4.")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if not quiet:
            print(" Done.", flush=True)

        # Reuse the same allocated memory for a small speedup.
        self.capture_double_buffer = None

    def _get_frame(self):
        ret, self.capture_double_buffer = self.cap.read(self.capture_double_buffer)
        timestamp = self.get_current_timestamp()
        frame, full_frame = None, None

        if self.capture_double_buffer is not None:
            full_frame = self.capture_double_buffer.copy()
            frame = self.subsample_frame(full_frame, normalize=True)

        return ret, frame, full_frame, timestamp

    def stop(self):
        if self.cap is not None:
            self.cap.release()

class Flea3Capture(Camera):
    def __init__(
        self, height, width, fps, device, subsample=0,  fullframe_path=None, gain=100, cam_identifier=None, start_timestamp=None, roi=None, **kwargs
    ):
        super().__init__(height, width, fps=fps, subsample=subsample, fullframe_path=fullframe_path, cam_identifier=cam_identifier, start_timestamp=start_timestamp, roi=roi, **kwargs)

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
        full_frame = np.array(self.cap.retrieveBuffer())
        timestamp = self.get_current_timestamp()

        im = self.subsample_frame(full_frame, normalize=True)

        return True, im, full_frame, timestamp

class Flea3CapturePySpin(Camera):
    def __init__(
        self, height, width, fps, device, subsample=0,  fullframe_path=None, gain=100, cam_identifier=None, start_timestamp=None, roi=None, **kwargs
    ):
        super().__init__(height, width, fps=fps, subsample=subsample, fullframe_path=fullframe_path, cam_identifier=cam_identifier, start_timestamp=start_timestamp,
                        roi=roi, **kwargs)

        self.device = device
        self.camera = None
        try:
            # using a 'with' statement initializes it and closes it, and so will fail if it doesn't actually exist
            with simple_pyspin.Camera(index=str(self.device)) as cam:
                print('camera at',self.device)
            self.camera = simple_pyspin.Camera(index=str(self.device))
        except:
            print("Could not open camera by serial number.. Trying by index.")
            self.camera = None

        if self.camera is None:
            # open by index if serial number fails
            try:
                with simple_pyspin.Camera(index=int(self.device)) as cam:
                    print('camera',self.device,'found')
            except:
                raise RuntimeError(f"Failed to initialize camera at {self.device}")                
            self.camera = simple_pyspin.Camera(index=int(self.device))

        # settings need to go before initialization
        
        self.camera.PixelFormat = "Mono8"
        self.camera.AcquisitionFrameRateEnable = True
        self.camera.AcquisitionFrameRate = int(fps)
        #self.camera.GammaEnabled = False
        self.camera.init()

        self.camera.start()

    def _get_frame(self):
        full_frame = self.camera.get_array()
        timestamp = self.get_current_timestamp()

        im = self.subsample_frame(full_frame, normalize=True)

        return True, im, full_frame, timestamp

    def stop(self):
        self.camera.stop()
        self.camera.close()

def cam_generator(cam_object, warmup=True, *args, **kwargs):
    cam = cam_object(*args, **kwargs)
    if warmup:
        cam.warmup()

    try:
        while True:
            ret, frame, frame_orig, timestamp = cam.get_frame()
            if not ret:
                break
            yield ret, frame, frame_orig, timestamp
    finally:
        if cam is not None:
            cam.stop()
