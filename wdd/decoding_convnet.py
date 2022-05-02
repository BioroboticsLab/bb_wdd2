import cv2
import numpy as np
import numba
import scipy.signal
import scipy.stats
import skimage.filters
import sklearn.linear_model

import torch
import torch.nn
import torch.nn.functional

from bb_wdd_filter.models_supervised import WDDClassificationModel, DEFAULT_CLASS_LABELS
from bb_wdd_filter.dataset import ImageNormalizer

class WaggleDecoderConvNet():

    def __init__(self, fps, bee_length, model_path,
                image_size=32,
                temporal_dimension=40,
                expected_bee_length=44.0,
                default_scale_factor=0.5,
                cuda=True):
        """
            Arguments:
                temporal_dimension, expected_bee_length, default_scale_factor (int, float, float)
                Parameters the loaded checkpoint was trained with. Only change if you retrain the network with new data.
        """
        self.fps = fps
        self.bee_length = bee_length
        self.expected_bee_length = expected_bee_length
        self.temporal_dimension = temporal_dimension
        self.device = torch.device("cpu" if not cuda else "cuda")

        bee_length_factor = (bee_length / expected_bee_length)
        scale_factor = default_scale_factor / bee_length_factor
        print("Decoder scaling images by {:3.2f} and cropping to {}x{}.".format(
            scale_factor, image_size, image_size))

        self.model = WDDClassificationModel(image_size=image_size)
        self.normalizer = ImageNormalizer(image_size=image_size, scale_factor=scale_factor)

        if cuda:
            self.model.cuda()
        self.model.eval()

        state = torch.load(model_path)
        self.model.load_state_dict(state["model"])

    def predict_on_images(self, images):
        n_images = len(images)

        if n_images > 2 * self.temporal_dimension:
            # If possible, take images starting from the center of the recorded images.
            # They are a bit more likely to be in the waggle than images directly at the start or end.
            start_index = n_images // 2
            end_index = start_index + self.temporal_dimension
            images = images[start_index:end_index]

        images = images[-self.temporal_dimension:]
        images = self.normalizer.normalize_images(images)

        images = [torch.from_numpy(i).to(self.device, non_blocking=True) for i in images]
        images = torch.stack(images, dim=0)

        images = images.unsqueeze(0) # Add channel dimension.
        images = images.unsqueeze(0) # Add batch dimension.

        predictions = self.model(images)
        assert predictions.shape[2] == 1
        assert predictions.shape[3] == 1
        predictions = predictions[:, :, 0, 0, 0]

        classes, vectors, durations, confidences = self.model.postprocess_predictions(predictions, return_raw=False, as_numpy=True)

        assert classes.shape[0] == 1 # batch size == 1

        return (
            classes[0],
            vectors[0],
            durations[0],
            confidences[0]
        )
        

    def __call__(self, waggle, full_frame_rois, metadata_dict, **kwargs):
        with torch.no_grad():
            predicted_class, vectors, duration, confidence = self.predict_on_images(full_frame_rois)

        # Make the duration serializable to JSON.
        if np.isnan(duration):
            duration = None

        metadata_dict["predicted_class"] = int(predicted_class)
        metadata_dict["predicted_class_label"] = DEFAULT_CLASS_LABELS[predicted_class]

        metadata_dict["predicted_class_confidence"] = float(confidence)
        metadata_dict["waggle_angle"] = float(np.arctan2(vectors[1], vectors[0]))
        metadata_dict["waggle_duration"] = float(duration)

        return waggle, full_frame_rois, metadata_dict, kwargs
