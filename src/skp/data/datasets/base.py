import cv2
import glob
import random
import numpy as np
import os, os.path as osp
import re
import torch

from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

from . import helper


class ImageDataset(Dataset):
    """
    Dataset to be used for conventional 8-bit images (e.g., PNG, JPEG).
    """
    def __init__(self,
                 inputs,
                 labels,
                 resize=None,
                 augment=None,
                 crop=None,
                 preprocess=None,
                 channels="bgr",
                 flip=False,
                 transpose=False,
                 invert=False,
                 reverse_channels=False,
                 verbose=True,
                 test_mode=False,
                 return_name=False,
                 return_imsize=False):

        self.inputs = inputs
        self.labels = labels
        self.resize = resize
        self.augment = augment
        self.crop = crop
        self.preprocess = preprocess
        assert channels.lower() in ["rgb", "bgr", "grayscale",
                                    "2dc"], f"{channels} is not a valid argument for `channels`"
        self.channels = channels.lower()

        # Always turned off if test_mode=False
        self.flip = flip
        self.transpose = transpose
        self.invert = invert
        self.reverse_channels = reverse_channels

        self.verbose = verbose
        self.test_mode = test_mode
        self.return_name = return_name
        self.return_imsize = return_imsize

    def __len__(self):
        return len(self.inputs)

    def process_image(self, data):

        # Assumes channels-LAST for everything
        if not self.test_mode:
            if self.invert:
                data = helper.invert(data)
            if self.reverse_channels:
                data = helper.reverse_channels(data)
            if self.flip:
                data = helper.flip_array_2d(data)

        if self.resize:
            data = self.resize(**data)
        if self.transpose and not self.test_mode:
            data = helper.transpose_2d(data)
        if self.augment and not self.test_mode:
            data = self.augment(**data)
        if self.crop:
            data = self.crop(**data)
        if self.preprocess:
            data["image"] = self.preprocess(data["image"])

        # Change to channels-FIRST
        for k, v in data.items():
            if k == "image":
                # Image ALWAYS has channel dimension
                data[k] = np.ascontiguousarray(v.transpose(2, 0, 1))
            elif k == "mask":
                # Mask may not
                if v.ndim == 3:
                    data[k] = v.transpose(2, 0, 1)

        return data

    def load_image(self, i):
        if self.channels in ["bgr", "rgb"]:
            X = cv2.imread(self.inputs[i])
            # OpenCV reads images as BGR
            if self.channels == "rgb":
                X = np.ascontiguousarray(X[:, :, ::-1])
        elif self.channels == "grayscale":
            X = cv2.imread(self.inputs[i], 0)
            if isinstance(X, np.ndarray):
                X = np.expand_dims(X, axis=-1)
        elif self.channels == "2dc":
            # The input will be a string of filenames, separated by commas
            inputs = self.inputs[i].split(",")
            inputs = [cv2.imread(each_input, 0) for each_input in inputs]
            X = np.concatenate([np.expand_dims(each_input, axis=-1) for each_input in inputs], axis=-1)
        return X

    def get(self, i):
        try:
            X = self.load_image(i)

            if not isinstance(X, np.ndarray):
                print(f"OpenCV failed to load {self.inputs[i]} and returned `None`")
                return None

            return {"image": X}

        except Exception as e:
            if self.verbose:
                print(e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while isinstance(data, type(None)):
            if self.verbose:
                print("Failed to read {} !".format(self.inputs[i]))
            i = np.random.randint(len(self))
            data = self.get(i)

        imsize = data["image"].shape[:2]

        data = self.process_image(data)

        X = data["image"]
        del data

        X = torch.tensor(X).float()
        y = torch.tensor(self.labels[i]).float()

        out = [X, y]
        if self.return_name:
            out.append(self.inputs[i])
        if self.return_imsize:
            out.append(imsize)

        return tuple(out)


class ImageSegmentDataset(ImageDataset):
    """
    Dataset for segmentation problems, where the input is a conventional 8-bit image.

    For this dataset, there should be 1 segmentation label file for each input. This can be
    in varying formats.
    """
    def __init__(self, *args, **kwargs):
        self.segmentation_format = kwargs.pop("segmentation_format", "png")
        self.add_foreground_channel = kwargs.pop("add_foreground_channel", False)
        self.one_hot_encode = kwargs.pop("one_hot_encode", False)
        # IMPORTANT: num_classes does NOT include background class
        self.num_classes = kwargs.pop("num_classes", None)
        if self.one_hot_encode: assert self.num_classes is not None

        # Only really used for PNG/JPG format
        self.max_255 = kwargs.pop("max_255", False)

        assert bool(re.search(r"png|jpg|jpeg|npy|numpy|multislice_pred", self.segmentation_format))
        super().__init__(*args, **kwargs)

    def load_segmentation(self, i):
        if self.labels[i] == 0:
            # This means we are doing inference and don't need segmentations
            return np.zeros((512, 512, 1))
        if self.segmentation_format in ["png", "jpeg", "jpg"]:
            assert self.num_classes is not None
            y = cv2.imread(self.labels[i], cv2.IMREAD_UNCHANGED)
            # In case there are fewer than 3 classes
            y = y[..., :self.num_classes].astype("float")
            if self.max_255:
                y /= 255
        elif self.segmentation_format in ["npy", "numpy"]:
            y = np.load(self.labels[i])
        elif self.segmentation_format == "multislice_pred":
            # This is when the input is 2Dc, and you are trying to predict
            # the segmentation label for each channel
            fp = self.labels[i].split(",")
            if self.test_mode:
                # During inference, the label will just be the CENTER
                y = cv2.imread(fp[len(fp)//2])
            else:
                y = [cv2.imread(_) for _ in fp]
                y = np.concatenate(y, axis=-1)
        else:
            raise Exception(f"{self.segmentation_format} label format is not supported")

        if self.add_foreground_channel:
            if y.shape[-1] == 1:
                print("add_foreground_channel=True, however only 1 class present, so this will be ignored")
            else:
                fg = np.expand_dims(y.sum(-1), -1)
                fg[fg > 0] = 1

        return y

    def get(self, i):
        try:
            X = self.load_image(i)

            if not isinstance(X, np.ndarray):
                print(f"OpenCV failed to load {self.inputs[i]} and returned `None`")
                return None

            y = self.load_segmentation(i)
            return {"image": X, "mask": y}

        except Exception as e:
            if self.verbose:
                print(e)
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while isinstance(data, type(None)):
            if self.verbose: print("Failed to read {} !".format(self.inputs[i]))
            i = np.random.randint(len(self))
            data = self.get(i)

        imsize = data["image"].shape[:2]

        data = self.process_image(data)
        X, y = data["image"], data["mask"]
        del data

        X = torch.tensor(X).float()
        y = torch.tensor(y).float()

        if self.one_hot_encode:
            assert y.ndim == X.ndim - 1
            # num_classes does NOT include background class
            # Thus must pass self.num_classes+1 and then ignore the first channel
            y = torch.nn.functional.one_hot(y.long(), num_classes=self.num_classes + 1)[..., 1:].permute(2, 0,
                                                                                                         1).float()
        out = [X, y]
        if self.return_name:
            out.append(self.inputs[i])
        if self.return_imsize:
            out.append(imsize)

        return tuple(out)
