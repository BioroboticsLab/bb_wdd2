import numpy as np
import cv2
from imageio import imread
from skimage.transform import resize
import time


try:
    import PyCapture2
except ImportError:
    print("Unable to import PyCapture2, Flea3 cameras won't work")
        
        
class Camera:
    def __init__(self, height, width, background=None, alpha=None):
        self.background = background
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = .99
        self.height = height
        self.width = width
        
    def _get_frame(self):
        raise NotImplementedError()
        
    def get_frame(self):
        ret, frame_orig = self._get_frame()
        
        if not ret:
            return ret, frame_orig, frame_orig
        
        #FIXME: temporary hack to speed up image aquisition using Flea3 in the BeesBook setup
        #frame = resize(frame_orig, (self.height, self.width), mode='constant', order=1, anti_aliasing=False)
        frame = frame_orig[::2, ::2]
        
        if self.background is None:
            self.background = np.copy(frame)
        else:
            self.background = self.alpha * self.background + (1 - self.alpha) * frame
            
        return ret, frame, frame_orig
    
    def warmup(self, tolerance=0.01, num_frames_per_round=100, num_hits=3):
        fps_target = self.fps - tolerance * self.fps
        
        print('Camera warmup, FPS target >= {:.1f}'.format(fps_target))
        
        hits = 0
        while True:
            frame_idx = 0
            start_time = time.time()
            
            for _ in range(num_frames_per_round):
                ret, _, _ = self.get_frame()
                assert ret
                
                frame_idx += 1
                
            end_time = time.time()
            processing_fps = frame_idx / (end_time - start_time)
            fps_target_hit = processing_fps > fps_target
            
            if fps_target_hit:
                hits += 1
            else:
                hits = 0
                
            print('FPS: {:.1f} [{}]'.format(processing_fps, fps_target_hit))    
            
            if hits >= num_hits:
                print('Success')
                break


class OpenCVCapture(Camera):
    def __init__(self, height, width, fps, device, background=None, alpha=None):
        super().__init__(height, width, background, alpha)
        
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
    def __init__(self, height, width, fps, device, background=None, alpha=None, gain=100):
        super().__init__(height, width, background, alpha)
        
        self.fps = fps
        
        bus = PyCapture2.BusManager()
        numCams = bus.getNumOfCameras()
        
        if not numCams:
            raise RuntimeError('No Flea3 camera detected')
        
        self.cap = PyCapture2.Camera()
        self.uid = bus.getCameraFromIndex(int(device))
        self.cap.connect(self.uid)
        
        fmt7imgSet = PyCapture2.Format7ImageSettings(PyCapture2.MODE.MODE_4, 0, 0, 2048, 1080, PyCapture2.PIXEL_FORMAT.MONO8)
        fmt7pktInf, isValid = self.cap.validateFormat7Settings(fmt7imgSet)
        if not isValid:
            raise RuntimeError("Format7 settings are not valid!")
        
        self.cap.setFormat7ConfigurationPacket(fmt7pktInf.recommendedBytesPerPacket, fmt7imgSet)
        
        self.cap.setProperty(type=PyCapture2.PROPERTY_TYPE.FRAME_RATE, absValue=fps)
        self.cap.setProperty(type=PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE, absValue=False)
        self.cap.setProperty(type=PyCapture2.PROPERTY_TYPE.SHUTTER, absValue=1/fps * 1000)
        self.cap.setProperty(type=PyCapture2.PROPERTY_TYPE.GAIN, absValue=100)

        self.cap.startCapture()
        
    def _get_frame(self, ramdisk_path=b'/home/ben/ramdisk/'):
        image = self.cap.retrieveBuffer()
    
        image.save(ramdisk_path, PyCapture2.IMAGE_FILE_FORMAT.BMP)
        im = imread(ramdisk_path, format='bmp')
        im = (im[::3, ::3].astype(np.float32) / 255) * 2 - 1

        return True, im


def cam_generator(cam_object, *args, **kwargs):
    cam = cam_object(*args, **kwargs)
    cam.warmup()

    while True:
        ret, frame, frame_orig = cam.get_frame()
        if not ret:
            break
        yield ret, frame, frame_orig, cam.background
