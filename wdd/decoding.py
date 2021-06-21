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


@numba.njit
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
    
class WaggleDecoder():

    def __init__(self, fps, bee_length):
        self.fps = fps
        self.bee_length = bee_length

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
        diff_images = map(lambda m: skimage.filters.gaussian(m, sigma=kernel_size), diff_images)
        diff_images = np.stack(list(diff_images))
        return diff_images

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
        kernel_size = fourier.shape[0]
        kernel = self.mexican_hat_2d(kernel_size, sigma1=int(6 * self.bee_length), sigma2=int(self.bee_length))
        fourier_kernel = np.abs(np.fft.fftshift(np.fft.fft2(kernel)))
        filtered = np.power(np.multiply(fourier, fourier_kernel), 5.0)
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

    def get_correlation_with_frequency(self, images, hz, samplerate):
        """Takes a 3D array of images (number, height, width) and calculates each pixels' activation with morlet wavelet corresponding to a certain frequency.

        Arguments:
            images: np.array shape number, height, width
            hz: float
                Frequency of the morlet wavelet.
            samplerate: float
                Samplerate of the original images.
        
        Returns:
            correlations: np.array shape number, height, width
        """
        s, w = 0.25, 15.0
        M = int(2*s*w*samplerate / hz)
        wavelet = scipy.signal.morlet(M, w, s, complete=False)
        wavelet = np.real(wavelet).astype(np.float32)
        wavelet /= np.abs(wavelet).max()

        wavelet = wavelet.reshape(wavelet.shape[0], 1, 1)

        corr = scipy.ndimage.convolve(images, wavelet)
        corr = np.abs(corr)
        return corr

    def estimate_waggle_begin_end(self, difference_images):
        """Takes a set of first-order differences images of a recording and estimates the begin and end of a contained waggle run.
        Also returns the regions of activity of the first and second half of the waggle..

        Arguments:
            difference_images: np.array shape n, height, width

        Returns:
            ((int, int), (np.array, np.array)) or ((None, None), None)
        """
        difference_images = difference_images - difference_images.mean()
        difference_images /= (np.abs(difference_images).max() + 0e-4)
        difference_images /= 3.0 # N frequencies

        corr = self.get_correlation_with_frequency(difference_images, 12, samplerate=self.fps) \
                + self.get_correlation_with_frequency(difference_images, 13, samplerate=self.fps) \
                + self.get_correlation_with_frequency(difference_images, 14, samplerate=self.fps)
        
        activations = np.mean(corr, axis=(1, 2))
        
        activations = scipy.ndimage.maximum_filter1d(activations, int(self.fps/6))
        t = skimage.filters.threshold_otsu(activations)

        a = activations > t
        begin, end = largest_consecutive_region(a)
        if begin is None:
            return (None, None), None
        corr = corr[begin:end]
        return (begin, end), corr

    def estimate_angle_from_moments(self, difference_images, waggle_regions):
        """Takes an array of first-order difference images and a heatmap of likely waggle regions.
        Calculates the centroids of the absolute differences for the first and last third of images weighted by the waggle regions.
        The vector between these centroids defines the most likely angle of motion.

        Arguments:
            difference_images: np.array shape n, height, width
            waggle_regions: tuple(np.array(height, width))

        Returns:
            angle: float
            Can be np.nan.
        """

        points = np.zeros(shape=(waggle_regions.shape[0], 3))
        main_waggle_region = np.mean(waggle_regions, axis=0)
        for idx in range(waggle_regions.shape[0]):
            img = waggle_regions[idx]
            
            kernel_size = self.bee_length / 2.0
            img = skimage.filters.gaussian(img * main_waggle_region, sigma=kernel_size)
            crop_width = int(kernel_size)
            img = img[crop_width:-crop_width, crop_width:-crop_width]

            cy, cx = np.unravel_index(np.argmax(img), img.shape)
            cmax = img[cy, cx]

            points[idx, :] = (cx, cy, cmax)


        if points.shape[0] > 4:
            points = points[points[:, 2] >= np.median(points[:, 2])]
        points = points[:, :2]
        first_half, second_half = points[:(points.shape[0] // 2)], points[(points.shape[0] // 2):]

        all_vectors = second_half[:, None, :] - first_half
        all_vectors = all_vectors.reshape(-1, 2)
        all_vectors /= np.linalg.norm(all_vectors, axis=0)
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
        (waggle_begin, waggle_end), waggle_regions = self.estimate_waggle_begin_end(difference_images)
        waggle_length = np.nan
        if waggle_begin is not None:
            difference_images = difference_images[waggle_begin:waggle_end]
            waggle_length = waggle_end - waggle_begin
        fourier_sum = self.accumulate_fourier_transform(difference_images)
        filtered_fourier = self.filter_fft_with_kernel(fourier_sum)
        fourier_angle = self.calculate_angle_from_filtered_fourier(filtered_fourier)
        
        positional_angle = self.estimate_angle_from_moments(difference_images, waggle_regions)
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
