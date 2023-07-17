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
                n_batches=1,
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
        self.n_batches = n_batches
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
        
        batch_size = max(1, min(self.n_batches, n_images // self.temporal_dimension))

        def get_sequence_from_images(offset_step):
            begin = n_images // 2 + (offset_step - (batch_size // 2)) * self.temporal_dimension
            end = begin + self.temporal_dimension
            if begin < 0 or end > n_images:
                return None
            
            sequence = images[begin:end]
            sequence = self.normalizer.normalize_images(sequence)

            sequence = [torch.from_numpy(i).to(self.device, non_blocking=True) for i in sequence]
            sequence = torch.stack(sequence, dim=0)
            sequence = sequence.unsqueeze(0) # Add channel dimension.

            return sequence

        sequences = [get_sequence_from_images(i) for i in range(batch_size)]
        sequences = [s for s in sequences if s is not None]
        images = torch.stack(sequences, axis=0)

        predictions = self.model(images)
        assert predictions.shape[0] == len(sequences)
        assert predictions.shape[2] == 1
        assert predictions.shape[3] == 1
        predictions = predictions[:, :, 0, 0, 0]

        classes, vectors, durations, confidences = self.model.postprocess_predictions(predictions, return_raw=False, as_numpy=True)
        assert classes.shape[0] == len(sequences)
        # Some postprocessing to get the most confident non-other classification.
        most_certain_sequence = 0
        specific_predictions = classes != 0
        if np.any(specific_predictions):
            conf = confidences.copy()
            conf[~specific_predictions] /= 2.0
            most_certain_sequence = np.argmax(conf)

        waggle_detections = classes == 1
        if np.any(waggle_detections):
            mean_vectors = np.mean(vectors[waggle_detections], axis=0)
            vectors[waggle_detections] = mean_vectors[None, :]

        return (
            classes[most_certain_sequence],
            vectors[most_certain_sequence],
            durations[most_certain_sequence],
            confidences[most_certain_sequence]
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
