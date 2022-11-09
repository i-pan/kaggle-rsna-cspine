import numpy as np
import torch


class Preprocessor(object):
    """
    Object to deal with preprocessing.
    Easier than defining a function.
    """
    def __init__(self,
                 image_range,
                 input_range,
                 mean,
                 sdev,
                 channels_last=True):
        self.image_range = image_range
        self.input_range = input_range
        self.mean = mean
        self.sdev = sdev
        self.channels_last = channels_last
        self.num_channels = None

    def __call__(self, img):
        if not isinstance(self.num_channels, (int, float)):
            self.num_channels = img.shape[-1] if self.channels_last else img.shape[0]

        if isinstance(img, np.ndarray):
            img = img.astype("float")
        elif isinstance(img, torch.Tensor):
            img = img.float()

        # Preprocess an input image
        image_min = float(self.image_range[0])
        image_max = float(self.image_range[1])
        model_min = float(self.input_range[0])
        model_max = float(self.input_range[1])
        image_range = image_max - image_min
        model_range = model_max - model_min
        img = (((img - image_min) * model_range) / image_range) + model_min

        assert len(self.mean) == len(self.sdev)
        assert len(self.mean) in [self.num_channels, 1], "Number of image normalization parameters must match number " \
                                                         "of channels or equal 1"

        if len(self.mean) == self.num_channels:
            for channel in range(self.num_channels):
                if self.channels_last:
                    img[..., channel] -= self.mean[channel]
                    img[..., channel] /= self.sdev[channel]
                else:
                    img[channel] -= self.mean[channel]
                    img[channel] /= self.sdev[channel]
        else:
            img -= self.mean[0]
            img /= self.sdev[0]

        return img

    def denormalize(self, img):
        if len(self.mean) == self.num_channels:
            for channel in range(self.num_channels):
                if self.channels_last:
                    img[..., channel] *= self.sdev[channel]
                    img[..., channel] += self.mean[channel]
                else:
                    img[channel] *= self.sdev[channel]
                    img[channel] += self.mean[channel]
        else:
            img *= self.sdev[0]
            img += self.mean[0]

        image_min = float(self.image_range[0])
        image_max = float(self.image_range[1])
        model_min = float(self.input_range[0])
        model_max = float(self.input_range[1])
        image_range = image_max - image_min
        model_range = model_max - model_min

        img = ((img - model_min) * image_range) / model_range + image_min
        return img
