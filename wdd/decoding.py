"""
    This file is a port and improvement of the code used for the WDD2017 paper.
    https://github.com/BioroboticsLab/WDD_paper/blob/master/OrientationModule/FourierAngle.py

    Original algorithm by F. Wario.

    The entry point is the function calculate_fourier_angle_from_waggle_run_images(positions, images).
"""

import cv2
import numpy as np
import numba
import scipy.signal
import scipy.stats
import skimage.filters
import sklearn.linear_model


@numba.njit(cache=True, nogil=True)
def largest_consecutive_region(a):
    """Calculate the largest consecutive region of a boolean array.

    Arguments:
        a: np.array
            boolean type.

    Returns:
        begin, end: int
            Inclusive begin and exclusive end index.
    """
    begin, end = None, None
    max_length = None
    current_begin = None
    current_length = 0
    
    for i in range(a.shape[0]+1):
        active = a[i] if i < a.shape[0] else False
        if active:
            if current_begin is None:
                current_begin = i
            current_length += 1
            continue
        
        if (current_begin is not None) and ((max_length is None) or (current_length > max_length)):
            max_length = current_length
            begin, end = current_begin, current_begin + current_length
        current_length = 0
        current_begin = None
    return begin, end

def debug_display_img(img, delay=1, vmin=None, vmax=None, angles=None, point=None):
    import cv2
    if vmin is None:
        vmin = img.min()
    activity_im = (img - vmin)
    if vmax is None:
        vmax = activity_im.max()
    activity_im /= vmax
    activity_im = (activity_im * 255.0).astype(np.uint8)
    activity_im = cv2.applyColorMap(activity_im, cv2.COLORMAP_VIRIDIS)
    if angles is not None:
        for idx, angle in enumerate(angles):
            if angle is None:
                continue
            color = [(120, 255, 0), (255, 120, 50)][idx]
            w = activity_im.shape[0] // 2
            mx, my = activity_im.shape[1] // 2, activity_im.shape[0] // 2
            # In this coordinate system, 0° is straight-right, positive is counter-clockwise.
            cv2.line(activity_im, 
                    (int(mx + w * np.cos(angle)), int(my + -1.0 * w * np.sin(angle))),
                    (int(mx - 0.5 * w * np.cos(angle)), int(my - 0.5 * -1.0 * w * np.sin(angle))),
                    color, thickness=2)
    if point is not None:
        cv2.circle(activity_im, point, 5, color=(255, 0, 0))
    cv2.imshow("WDD", activity_im)
    cv2.waitKey(int(delay))
class WaggleDecoder():

    def __init__(self, fps, bee_length):
        self.fps = fps
        self.bee_length = bee_length
        self.scaled_bee_length = 20.0
        if self.bee_length > self.scaled_bee_length:
            self.rescale_factor = self.scaled_bee_length / self.bee_length
        else:
            self.rescale_factor = 1.0
            self.scaled_bee_length = self.bee_length

        self.fourier_filtering_kernel = None
        self.wavelets = None

    def create_difference_images(self, images):
        """
            Arguments:
                images: np.array of shape (N, W, H) where N = number of images.
            Returns:
                diff_images: smoothed first order difference of images
        """
        assert images.shape[0] > 0
        assert images.shape[1] == images.shape[2]
        
        diff_images = np.diff(images, axis=0)
        assert diff_images.shape[0] == images.shape[0] - 1
        assert diff_images.shape[1] == images.shape[1]

        kernel_size = int(self.bee_length / 8.0)
        filter_output = np.zeros(shape=(diff_images.shape[1], diff_images.shape[2], diff_images.shape[0]), dtype=np.float32)
        for i in range(diff_images.shape[0]):
            skimage.filters.gaussian(diff_images[i], sigma=kernel_size, output=filter_output[:, :, i])

        if self.rescale_factor < 1.0:
            filter_output = skimage.transform.rescale(filter_output,
                            scale=self.rescale_factor,
                            anti_aliasing=False,
                            multichannel=True)
        filter_output = np.moveaxis(filter_output, 2, 0)

        return filter_output

    def accumulate_fourier_transform(self, diff_images):
        """Applies the fast fourier transform over a 3D matrix of images
        and centers the zero frequency to the middle. Returns the sum over all fourier images.
        
        Arguments:
            diff_images: np.array of shape (N, W, H) where N is the number of images.
        Returns:
            fourier: np.array of shape (W, H) containing the sum of the shifted fourier transforms.
        """
        # Apply fourier transform over last two axes.
        fouriers = np.abs(np.fft.fft2(diff_images, axes=(-2, -1)))
        # Shift the zero frequency into the center of the individual fourier images.
        fouriers = np.fft.fftshift(fouriers, axes=(-2, -1))
        return np.sum(fouriers, axis=0)

    def gaussian_kernel_2d(self, kernel_size, sigma):
        """Generates an image of shape (kernel_size, kernel_size) that contains a 2d gaussian distribution.

        Arguments:
            kernel_size: float
                Width and height of the resulting image.
            sigma: float
                Variance of the distribution.
        """
        x = np.arange(0, kernel_size)
        y = np.arange(0, kernel_size)
        x, y = np.meshgrid(x, y)
        pos = np.empty(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y
        k = scipy.stats.multivariate_normal.pdf(pos,
                                                mean=(kernel_size//2, kernel_size//2),
                                                cov=np.ones(2) * sigma)
        return k

    def mexican_hat_2d(self, kernel_size, sigma1=30, sigma2=4):
        """Generates an image of shape (kernel_size, kernel_size) that contains a mexican hat kernel.
        The kernel is generated as the difference between two gaussians.

        Arguments:
            kernel_size: float
                Width and height of the resulting image.
            sigma1, sigma2: float
                Variances of the two subtracted gaussian kernels.
        
        """
        k =  self.gaussian_kernel_2d(kernel_size+1, sigma=sigma1) - self.gaussian_kernel_2d(kernel_size+1, sigma=sigma2)
        k = k[0:kernel_size, 0:kernel_size]
        return k

    def filter_fft_with_kernel(self, fourier):
        """Filters an image in fourier space with a mexican hat kernel.
        The filtering is done by multiplying the frequency space images.

        Arguments:
            fourier: np.array of shape (W, H) with 2D fourier space.
        Returns:
            filtered: np.array of shape (W, H)
        """
        assert fourier.shape[0] == fourier.shape[1] # Not 100% sure if necessary.

        if self.fourier_filtering_kernel is None:
            kernel_size = fourier.shape[0]
            self.fourier_filtering_kernel = self.mexican_hat_2d(kernel_size, sigma1=int(3 * self.scaled_bee_length), sigma2=int(0.5 * self.scaled_bee_length))
            self.fourier_filtering_kernel = np.abs(np.fft.fftshift(np.fft.fft2(self.fourier_filtering_kernel)))

        filtered = np.power(np.multiply(fourier, self.fourier_filtering_kernel), 5.0)
        return filtered

    def calculate_angle_from_filtered_fourier(self, filtered_fourier):
        """Takes a 2d fourier transform that has been filtered by a mexican hat kernel.
        Calculates the angle defined by the image moments.
        
        Arguments:
            filtered_fourier: np.array of shape (W, H) containing a spectral image that has been filtered.
        Returns:
            angle: Angle estimation of the underlying image movement. Note that the angle might be flipped.
        """
        image_moments = cv2.moments(filtered_fourier)
        image_covariance = np.array([[image_moments['nu20'], image_moments['nu11']],
                                    [image_moments['nu11'], image_moments['nu02']]])
        w, v = np.linalg.eig(image_covariance)
        eigenvector_idx = np.argmin(w)
        angle = -np.arctan2(v[1][eigenvector_idx], v[0][eigenvector_idx])
        return angle

    def get_wavelet(self, hz, samplerate):
        """Calculates a short morlet wavelet that corresponds to a given frequency at a given samplerate.

        Arguments:
            hz: float
                Frequency of the morlet wavelet.
            samplerate: float
                Samplerate of the original images.
        """
        s, w = 0.25, 15.0
        M = int(2*s*w*samplerate / hz)
        wavelet = scipy.signal.morlet(M, w, s, complete=False)
        wavelet = np.real(wavelet).astype(np.float32)
        wavelet /= np.abs(wavelet).max()

        wavelet = wavelet.reshape(wavelet.shape[0], 1, 1)

        return wavelet

    def get_correlation_with_wavelet(self, images, wavelet):
        """Takes a 3D array of images (number, height, width) and calculates each pixels' activation with morlet wavelet corresponding to a certain frequency.

        Arguments:
            images: np.array shape number, height, width
            wavelet: np.array shape d, 1, 1
                Where d < number
        
        Returns:
            correlations: np.array shape number, height, width
        """

        corr = scipy.ndimage.convolve(images, wavelet)
        corr = np.abs(corr)
        return corr

    def estimate_waggle_begin_end(self, images):
        """Takes a set of images of a recording and estimates the begin and end of a contained waggle run.
        Also the wavelet response during that central region.

        Arguments:
            images: np.array shape n, height, width

        Returns:
            ((int, int), (np.array)) or ((None, None), None)
        """
        
        if self.wavelets is None:
            self.wavelets = []
            for freq in (12, 13, 14):
                wavelet = self.get_wavelet(freq, self.fps)
                self.wavelets.append(wavelet)
        
        images = np.moveaxis(images, 0, 2)
        if self.rescale_factor < 1.0:
            images = skimage.transform.rescale(images,
                            scale=self.rescale_factor,
                            anti_aliasing=False,
                            multichannel=True)
        images = np.moveaxis(images, 2, 0)

        images = images - images.mean(axis=0)
        images /= (np.abs(images).max(axis=0) + 0e-4)
        images /= len(self.wavelets)

        corr = None
        for wavelet in self.wavelets:
            wavelet_corr = self.get_correlation_with_wavelet(images, wavelet)
            if corr is None:
                corr = wavelet_corr
            else:
                corr += wavelet_corr

        main_waggle_region = np.mean(corr, axis=0)
        main_waggle_region = skimage.filters.gaussian(main_waggle_region, sigma=self.scaled_bee_length / 2.0)
        corr = corr * main_waggle_region
        # Only calculate activation based on center of ROI to remove border artefacts and reduce overall noise.
        center = np.array(corr[0].shape) / 2.0
        assert center[0] > 2.0 * self.scaled_bee_length
        y0, y1 = int(center[0] - 1.5 * self.scaled_bee_length), int(center[0] + 1.5 * self.scaled_bee_length)
        x0, x1 = int(center[1] - 1.5 * self.scaled_bee_length), int(center[1] + 1.5 * self.scaled_bee_length)
        activations = np.mean(corr[:, y0:y1, x0:x1], axis=(1, 2))

        activations = scipy.ndimage.maximum_filter1d(activations, int(self.fps/6))
        if np.any(np.isnan(activations)):
            print("Warning: Invalid activation values: {}".format(activations))
            return (None, None), None
            
        t = skimage.filters.threshold_otsu(activations)

        a = activations > t
        begin, end = largest_consecutive_region(a)
        if begin is None:
            return (None, None), None
        corr = corr[begin:end]
        return (begin, end), corr

    def estimate_angle_from_moments(self, waggle_regions):
        """Takes a heatmap of likely waggle regions.
        Calculates the centroids of the waggle frequency response for each image.
        The vector between these centroids defines the most likely angle of motion.

        Arguments:
            waggle_regions: np.array(N, height, width)

        Returns:
            angle: float
            Can be np.nan.
        """

        points = np.zeros(shape=(waggle_regions.shape[0], 3))
        kernel_size = self.scaled_bee_length / 2.0
        # Determine the most central activity peak - likely the waggle.
        # Do this in order to not draw the image moments outwards too much.
        main_waggle_region = np.mean(waggle_regions, axis=0)
        main_waggle_region = skimage.filters.gaussian(main_waggle_region, sigma=kernel_size)
        peaks = skimage.feature.peak_local_max(main_waggle_region)
        if peaks.shape[0] > 0:
            image_center = np.array(waggle_regions[0].shape) / 2.0
            offset = np.linalg.norm(peaks - image_center, axis=1)
            middle_peak = np.argmin(offset)
            center = peaks[middle_peak]
            # Figure out small central region around waggle.
            y0, y1 = max(0, int(center[0] - self.scaled_bee_length)), min(waggle_regions.shape[1], int(center[0] + self.scaled_bee_length + 1))
            x0, x1 = max(0, int(center[1] - self.scaled_bee_length)), min(waggle_regions.shape[2], int(center[1] + self.scaled_bee_length + 1))
        
            waggle_regions = waggle_regions[:, y0:y1, x0:x1]
            
        filter_output = np.zeros(shape=waggle_regions[0].shape, dtype=np.float32)

        for idx in range(waggle_regions.shape[0]):
            img = waggle_regions[idx]
            
            skimage.filters.gaussian(img, sigma=kernel_size, output=filter_output)

            crop_width = int(kernel_size)
            img = filter_output[crop_width:-crop_width, crop_width:-crop_width]

            cy, cx = np.unravel_index(np.argmax(img), img.shape)
            cmax = img[cy, cx]

            points[idx, :] = (cx, cy, cmax)


        if points.shape[0] > 4:
            points = points[points[:, 2] >= np.median(points[:, 2])]
        else:
            if points.shape[0] < 2:
                return None
        points = points[:, :2]
        first_half, second_half = points[:(points.shape[0] // 2)], points[(points.shape[0] // 2):]

        all_vectors = second_half[:, None, :] - first_half
        all_vectors = all_vectors.reshape(-1, 2)
        norms = np.linalg.norm(all_vectors, axis=1)
        valid = norms > 0e-3
        all_vectors = all_vectors[valid, :] / norms[valid, None]

        ransac = sklearn.linear_model.RANSACRegressor()
        try:
            ransac.fit(all_vectors[:, :1], all_vectors[:, 1])
            direction = np.median(all_vectors[ransac.inlier_mask_], axis=0)
        except ValueError as e:
            return None
        # Image coordinate system is upside down.
        angle = -np.arctan2(direction[1], direction[0])
        return angle

    def correct_angle_direction(self, main_angle, orientation_angle):    
        """Flips an angle if the direction differs more than 90° from a baseline angle.
        
        Arguments:
            main_angle: Angle that is to be flipped.
            orientation_angle: Angle that defines the direction into which main_angle will be flipped.
        Returns:
            angle: main_angle or main_angle - np.pi
        """
        # Angles are converted to vectors.
        main = [np.cos(main_angle), np.sin(main_angle)]
        orientation = [np.cos(orientation_angle), np.sin(orientation_angle)]    
        # If the dot product is negative, the angles differ more than 90°.
        if np.dot(main, orientation) < 0.0:
            return main_angle - np.pi
        return main_angle

    def calculate_fourier_angle_from_waggle_run_images(self, images):
        """Calculates the angle of a waggle-run using the fourier transform of the difference images of successive frames.
        
        Arguments:
            images: np.array of shape (N, W, H) containing the cropped images around a waggle dance.
        """
        difference_images = self.create_difference_images(images)
        (waggle_begin, waggle_end), waggle_regions = self.estimate_waggle_begin_end(images)
        waggle_length = np.nan
        if waggle_begin is not None:
            difference_images = difference_images[waggle_begin:waggle_end]
            waggle_length = waggle_end - waggle_begin
        fourier_sum = self.accumulate_fourier_transform(difference_images)
        filtered_fourier = self.filter_fft_with_kernel(fourier_sum)
        fourier_angle = self.calculate_angle_from_filtered_fourier(filtered_fourier)
        
        positional_angle = None
        if waggle_regions is not None:
            positional_angle = self.estimate_angle_from_moments(waggle_regions)

        """for idx in range(images.shape[0]):
            angle = (fourier_angle, positional_angle)
            if waggle_begin is not None:
                if idx < waggle_begin or idx > waggle_end:
                    angle = None

            debug_display_img(images[idx], delay=40, angles=angle)
            """
        
        corrected_angle = fourier_angle
        if positional_angle is not None and not np.isnan(positional_angle):
            corrected_angle = self.correct_angle_direction(corrected_angle, positional_angle)
        corrected_angle = corrected_angle % (2*np.pi)
        return corrected_angle, waggle_length

    def __call__(self, waggle, full_frame_rois, metadata_dict, **kwargs):
        angle, length = self.calculate_fourier_angle_from_waggle_run_images(np.stack(full_frame_rois, axis=0))

        # Make the length serializable to JSON.
        if np.isnan(length):
            length = None
        else:
            length = length / self.fps
            
        metadata_dict["waggle_angle"] = angle
        metadata_dict["waggle_duration"] = length

        return waggle, full_frame_rois, metadata_dict, kwargs
