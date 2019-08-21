"""
    This file is a port and improvement of the code used for the WDD2017 paper.
    https://github.com/BioroboticsLab/WDD_paper/blob/master/OrientationModule/FourierAngle.py

    Original algorithm by F. Wario.

    The entry point is the function calculate_fourier_angle_from_waggle_run_images(positions, images).
"""

import cv2
import numpy as np
import scipy.stats

def create_difference_images(images):
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
    diff_images = np.apply_along_axis(lambda m: np.convolve(m, smoothing_kernel, mode="same"),
                                      axis=0, arr=diff_images)
    return diff_images

def accumulate_fourier_transform(diff_images):
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

def gaussian_kernel_2d(kernel_size, sigma):
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

def mexican_hat_2d(kernel_size, sigma1=30, sigma2=4):
    """Generates an image of shape (kernel_size, kernel_size) that contains a mexican hat kernel.
    The kernel is generated as the difference between two gaussians.

    Arguments:
        kernel_size: float
            Width and height of the resulting image.
        sigma1, sigma2: float
            Variances of the two subtracted gaussian kernels.
    
    """
    k =  gaussian_kernel_2d(kernel_size+1, sigma=sigma1) - gaussian_kernel_2d(kernel_size+1, sigma=sigma2)
    k = k[0:kernel_size, 0:kernel_size]
    return k

def filter_fft_with_kernel(fourier):
    """Filters an image in fourier space with a mexican hat kernel.
    The filtering is done by multiplying the frequency space images.

    Arguments:
        fourier: np.array of shape (W, H) with 2D fourier space.
    Returns:
        filtered: np.array of shape (W, H)
    """
    assert fourier.shape[0] == fourier.shape[1] # Not 100% sure if necessary.
    kernel_size = fourier.shape[0]
    kernel = mexican_hat_2d(kernel_size, sigma1=30, sigma2=4)
    fourier_kernel = np.abs(np.fft.fftshift(np.fft.fft2(kernel)))
    filtered = np.power(np.multiply(fourier, fourier_kernel), 5.0)
    return filtered

def calculate_angle_from_filtered_fourier(filtered_fourier):
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

def calculate_angle_from_positions(positions):
    """Takes a tensor of positions and calculates a robust angle.
    
    Arguments:
        positions: np.array of shape (N, 2) containing rows of (x, y) coordinates ordered by time.
    Returns:
        angle: Robust angle estimation of the movement.
    """
    # Make sure to disregard missing detections.
    positions = positions[~np.any(positions == -1, axis=1), :]
    assert np.all(positions != -1)
    
    n_half = positions.shape[0] // 2
    average_starting_point = positions[:n_half, :].mean(axis=0)
    # Calculate the mean of the different angles.
    directions = positions[(n_half+1):, :] - average_starting_point
    directional_angles = np.arctan2(directions[:, 0], directions[:, 1])
    # I doubt this is sensible.
    [n, c] = np.histogram(directional_angles, bins = 100)
    angle_modus = scipy.stats.circmean(c[np.argwhere(n == np.amax(n))])
    angle_modus_direction = np.array([np.cos(angle_modus), np.sin(angle_modus)])
    
    # Calculate the total mean direction.
    total_direction = positions[(n_half+1):, :].mean(axis=0) - average_starting_point
    
    # Find a compromise between the two directions.
    compromise_direction = angle_modus_direction + total_direction
    angle = -np.atan2(compromise_direction[1], compromise_direction[0])
    return angle

def correct_angle_direction(main_angle, orientation_angle):    
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

def calculate_fourier_angle_from_waggle_run_images(positions, images):
    """Calculates the angle of a waggle-run using the fourier transform of the difference images of successive frames.
    
    Arguments:
        positions: np.array of shape (N, 2) containing the x, y coordinates of the waggles.
        images: np.array of shape (N, W, H) containing the cropped images around a waggle dance.
    """
    difference_images = create_difference_images(images)
    fourier_sum = accumulate_fourier_transform(difference_images)
    filtered_fourier = filter_fft_with_kernel(fourier_sum)
    fourier_angle = calculate_angle_from_filtered_fourier(filtered_fourier)
    
    positional_angle = calculate_angle_from_positions(positions)
    
    corrected_angle = correct_angle_direction(fourier_angle, positional_angle)
    corrected_angle = corrected_angle % (2*np.pi)
    return corrected_angle
