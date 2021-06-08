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

    def __init__(self):
        pass

    def create_difference_images(self, images):
        """
            Arguments:
                images: np.array of shape (N, W, H) where N = number of images.
            Returns:
                diff_images: smoothed first order difference of images
        """
        assert images.shape[0] > 0
        assert images.shape[1] == images.shape[2]
        
        smoothing_kernel = np.ones(shape=(3, 3), dtype=np.float32) / 9.0
        diff_images = np.diff(images, axis=0)
        assert diff_images.shape[0] == images.shape[0] - 1
        assert diff_images.shape[1] == images.shape[1]

        diff_images = np.stack(list(map(lambda m: scipy.signal.convolve2d(m, smoothing_kernel, mode='same'), diff_images)))
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
        kernel = self.mexican_hat_2d(kernel_size, sigma1=30, sigma2=4)
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
        s, w = 0.5, 10.0
        M = int(2*s*w*samplerate / hz)
        wavelet = scipy.signal.morlet(M, w, s, complete=False)
        
        wavelet = wavelet.reshape(wavelet.shape[0], 1, 1)

        corr = np.real(scipy.ndimage.convolve(images, wavelet))
        corr2 = np.real(scipy.ndimage.convolve(images, -wavelet))
        corr = np.stack((corr, corr2))
        corr = np.max(corr, axis=0)
        
        return corr

    def estimate_waggle_begin_end(self, difference_images):
        """Takes a set of first-order differences images of a 60hz recording and estimates the begin and end of a contained waggle run.
        Also returns the regions of activity of the first and second half of the waggle..

        Arguments:
            difference_images: np.array shape n, height, width

        Returns:
            ((int, int), (np.array, np.array)) or ((None, None), None)
        """
        corr = self.get_correlation_with_frequency(difference_images, 12, samplerate=60) \
                + self.get_correlation_with_frequency(difference_images, 15, samplerate=60)

        activations = np.mean(corr, axis=(1, 2))
        activations /= np.abs(difference_images).mean()
        activations = scipy.ndimage.maximum_filter1d(activations, 10)
        t = skimage.filters.threshold_otsu(activations)

        t = 5.0 # 5% percentile of otsu thresholds of global population.
        a = activations > t
        begin, end = largest_consecutive_region(a)
        if begin is None:
            return (None, None), None
        length = end - begin
        first_half_location = np.mean(corr[begin:(begin+length//2)], axis=0) 
        second_half_location = np.mean(corr[(end-length//2):end], axis=0)
        return (begin, end), (first_half_location, second_half_location)

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
        length = difference_images.shape[0]
        if length < 6:
            return np.nan

        points = np.zeros(shape=(2, 2))
        N = length // 3
        for idx in range(2):
            if idx == 0:
                diff_images = difference_images[0:N]
            else:
                diff_images = difference_images[(length-N):]
            # Calculate the total motion activity in the image.
            diff_images = np.sum(np.abs(diff_images), axis=0)
            # And focus on the waggle region.
            if waggle_regions is not None:
                diff_images *= waggle_regions[idx]
            # Make image moment concentrate more on the highest activity regions.
            diff_images = np.power(diff_images, 5)
            moments = cv2.moments(diff_images)
            cx, cy = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
            points[idx, :] = (cx, cy)
            
        direction = points[1] - points[0]
        # Image coordinate system is upside down.
        angle = np.arctan2(-direction[1], direction[0])
        
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
        if not np.isnan(positional_angle):
            corrected_angle = self.correct_angle_direction(corrected_angle, positional_angle)
        corrected_angle = corrected_angle % (2*np.pi)
        return corrected_angle, waggle_length

    def __call__(self, waggle, full_frame_rois, metadata_dict, **kwargs):
        angle, length = self.calculate_fourier_angle_from_waggle_run_images(np.stack(full_frame_rois, axis=0))

        # Make the length serializable to JSON.
        if np.isnan(length):
            length = None

        metadata_dict["waggle_angle"] = angle
        metadata_dict["waggle_length"] = length

        return waggle, full_frame_rois, metadata_dict, kwargs
