import cv2
import glob
import numpy as np
import os.path as osp
import re
import torch

from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

from . import helper


class ImageStackDataset(Dataset):
    """
    Dataset to be used for 3D inputs where the input is assembled from
    2D 8-bit images (e.g., PNG, JPEG).

    Each element in `inputs` should be:
    a) DIRECTORY which contains all the images
    that will be used to construct the 3D input
    OR
    b) LIST of filenames separated by ",".

    The image filenames should be SORTABLE in ASCENDING ORDER in the desired order.
    """
    def __init__(self,
                 inputs,
                 labels,
                 num_images,
                 z_lt,
                 z_gt,
                 load_mode="directory",
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
        self.num_images = num_images
        self.load_mode = load_mode

        # These arguments specify how to deal with the z-axis dimension
        # when it is LESS THAN (z_lt) `num_images` and GREATER THAN (z_gt)
        # These are STRINGS in the format a_b where a specifies the TRAIN
        # strategy and b specifies the INFERENCE strategy
        assert z_lt in ["pad_pad", "resample_resample"]
        assert z_gt in ["crop_crop", "crop_resample", "resample_resample"]
        self.z_lt = z_lt
        self.z_gt = z_gt

        self.resize = resize
        self.augment = augment
        self.crop = crop
        self.preprocess = preprocess
        assert channels in ["rgb", "bgr", "grayscale"], f"{channels} is not a valid argument for `channels`"
        self.channels = channels

        # Only applied when test_mode=False
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

    @staticmethod
    def apply_op(func, data):
        num_img = len(data["image"])
        if "mask" in data:
            output = func(image=data["image"][0], mask=data["mask"][0],
                          **{f"image{i}": data["image"][i] for i in range(1, num_img)},
                          **{f"mask{i}": data["mask"][i] for i in range(1, num_img)})
            data["mask"] = np.concatenate([[output["mask"]] + [output[f"mask{i}"] for i in range(1, num_img)]])
        else:
            output = func(image=data["image"][0], **{f"image{i}": data["image"][i] for i in range(1, num_img)})

        data["image"] = np.concatenate([[output["image"]] + [output[f"image{i}"] for i in range(1, num_img)]])
        return data

    def process_image(self, data):
        # Currently, data is channels-LAST
        if not self.test_mode:
            if self.invert:
                data = helper.invert(data)
            if self.reverse_channels:
                data = helper.reverse_channels(data)
            if self.flip:
                data = helper.flip_array_3d(data)

        if self.resize:
            for k, v in data.items():
                data[k] = self.resize(v)
        if self.transpose and not self.test_mode:
            data = helper.transpose_3d(data)
        if self.augment and not self.test_mode:
            data = self.apply_op(self.augment, data)
        if self.crop:
            data = self.crop(data)
        if self.preprocess:
            data["image"] = self.preprocess(data["image"])

        for k, v in data.items():

            if isinstance(v, np.ndarray):
                if v.ndim == 4:
                    data[k] = np.ascontiguousarray(v.transpose(3, 0, 1, 2))
                elif v.ndim == 3:
                    data[k] = np.ascontiguousarray(v)
                else:
                    raise Exception(f"{k} does not have valid dimensions")

        return data

    @staticmethod
    def check_if_image(fp):
        return "png" in fp or "jpg" in fp or "jpeg" in fp

    def load_image(self, fp):
        if self.channels in ["bgr", "rgb"]:
            X = cv2.imread(fp)
            if self.channels == "rgb":
                X = np.ascontiguousarray(X[:, :, ::-1])
        elif self.channels == "grayscale":
            X = np.expand_dims(cv2.imread(fp, 0), axis=-1)
        return X

    def adjust_z_dimension(self, X):
        if self.test_mode:
            z_lt, z_gt = self.z_lt.split("_")[1], self.z_gt.split("_")[1]
        else:
            z_lt, z_gt = self.z_lt.split("_")[0], self.z_gt.split("_")[0]

        if len(X) > self.num_images:
            if z_gt == "resample":
                indices = zoom(np.arange(len(X)), float(self.num_images) / len(X), order=0,
                               prefilter=False)
                X = X[indices]
            elif z_gt == "crop":
                indices = np.arange(len(X))
                start_index = np.random.randint(0, len(X) - self.num_images)
                indices = indices[start_index:start_index+self.num_images]
                X = X[indices]
        elif len(X) < self.num_images:
            if z_lt == "resample":
                indices = zoom(np.arange(len(X)), float(self.num_images) / len(X), order=0,
                               prefilter=False)
                X = X[indices]
            elif z_lt == "pad":
                filler = np.stack([np.zeros_like(X[0])] * (self.num_images - len(X)))
                if np.min(X) < 0:
                    filler[...] = np.min(X)
                X = np.concatenate([X, filler])
        return X

    def get(self, i):
        try:
            if self.load_mode == "directory":
                slices = np.sort(glob.glob(osp.join(self.inputs[i], '*')))
                slices = [s for s in slices if self.check_if_image(s)]
            elif self.load_mode == "list":
                slices = self.inputs[i].split(",")

            if len(slices) == 0:
                print(f"No images found for {self.inputs[i]}")
                return None

            X = np.stack([self.load_image(s) for s in slices])
            X = self.adjust_z_dimension(X)
            assert len(X) == self.num_images
            return {"image": X}
        except Exception as e:
            if self.verbose:
                print(f'Failed to load {self.inputs[i]} :  {e}')
            return None

    def __getitem__(self, i):
        data = self.get(i)
        while isinstance(data, type(None)):
            if self.verbose:
                print("Failed to read {} !".format(self.inputs[i]))
            i = np.random.randint(len(self))
            data = self.get(i)

        imsize = data["image"].shape[:3]

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


class ImageStackSegmentDataset(ImageStackDataset):

    def __init__(self, *args, **kwargs):
        self.segmentation_format = kwargs.pop("segmentation_format", "png")
        self.add_foreground_channel = kwargs.pop("add_foreground_channel", False)
        self.one_hot_encode = kwargs.pop("one_hot_encode", False)
        # IMPORTANT: num_classes does NOT include background class
        self.num_classes = kwargs.pop("num_classes", None)
        if self.one_hot_encode:
            assert self.num_classes is not None

        # Only really used for PNG/JPG format
        self.max_255 = kwargs.pop("max_255", False)

        assert bool(re.search(r"png|jpg|jpeg|npy|numpy|multislice_pred", self.segmentation_format))
        super().__init__(*args, **kwargs)

    def load_segmentation(self, i):
        try:
            if self.labels[i] == 0:
                # This means we are doing inference and don't need segmentations, just return dummy
                return np.zeros((4, 128, 128, 1))
            if self.segmentation_format in ["png", "jpeg", "jpg"]:
                assert self.num_classes is not None
                seg_files = np.sort(glob.glob(osp.join(self.labels[i], "*")))
                y = np.stack([np.expand_dims(cv2.imread(each_seg), axis=0) for each_seg in seg_files])
                y = y[..., :self.num_classes].astype("float")
                if self.max_255:
                    y /= 255
            elif self.segmentation_format in ["npy", "numpy"]:
                # If numpy format, assume that the 3D segmentation volume is saved in
                # one file
                y = np.load(self.labels[i])
            else:
                raise Exception(f"{self.segmentation_format} label format is not supported")
            y = self.adjust_z_dimension(y)

            assert len(y) == self.num_images
            return y
        except Exception as e:
            if self.verbose:
                print(e)
            return None

    def load_input(self, i):
        try:
            slices = np.sort(glob.glob(osp.join(self.inputs[i], '*')))
            slices = [s for s in slices if self.check_if_image(s)]
            if len(slices) == 0:
                print(f"No images found for {self.inputs[i]}")
                return None

            X = np.stack([self.load_image(s) for s in slices])
            X = self.adjust_z_dimension(X)
            assert len(X) == self.num_images
            return X
        except Exception as e:
            if self.verbose:
                print(e)
            return None

    def get(self, i):
        X = self.load_input(i)
        y = self.load_segmentation(i)

        if not isinstance(X, np.ndarray):
            print(f"Failed to load {self.inputs[i]} and returned `None`")
            return None

        if not isinstance(y, np.ndarray):
            print(f"Failed to load {self.labels[i]} and returned `None`")
            return None

        # X and y should be in channels_last
        if not self.test_mode:
            assert X.shape[:3] == y.shape[
                                  :3], f"image dimensions {X.shape[:3]} do not match label dimensions {y.shape[:3]}"
        return {"image": X, "mask": y}

    def __getitem__(self, i):
        data = self.get(i)
        while isinstance(data, type(None)):
            if self.verbose:
                print("Failed to read {} !".format(self.inputs[i]))
            i = np.random.randint(len(self))
            data = self.get(i)

        imsize = data["image"].shape[:3]
        data = self.process_image(data)
        X, y = data["image"], data["mask"]
        del data

        X = torch.tensor(X).float()
        y = torch.tensor(y).float()

        if self.one_hot_encode:
            # num_classes does NOT include background class
            # Thus must pass self.num_classes+1 and then ignore the first channel
            y = torch.nn.functional.one_hot(y.long(), num_classes=self.num_classes + 1)
            y = y[..., 1:].permute(3, 0, 1, 2).float()

        out = [X, y]
        if self.return_name:
            out.append(self.inputs[i])
        if self.return_imsize:
            out.append(imsize)

        return tuple(out)


class NumpyChunkDataset(ImageStackDataset):

    def get(self, i):
        try:
            X = np.load(self.inputs[i])
            X = self.adjust_z_dimension(X)
            if X.ndim == 3:
                X = np.expand_dims(X, axis=-1)
            assert len(X) == self.num_images
            return {"image": X}
        except Exception as e:
            if self.verbose:
                print(f'Failed to load {self.inputs[i]} :  {e}')
            return None


class NumpyChunkSegmentDataset(ImageStackSegmentDataset):
    """
    This dataset assumes that chunks have been premade and saved to disk
    as Numpy arrays.
    """

    # def load_segmentation(self, i):
    #     assert self.segmentation_format in ["npy", "numpy"]
    #     if self.labels[i] == 0:
    #         # This means we are doing inference and don't need segmentations, return dummy
    #         return np.zeros((4, 128, 128, 1))
    #     y = np.load(self.labels[i])
    #     return y

    def load_input(self, i):
        try:
            X = np.load(self.inputs[i])
            X = self.adjust_z_dimension(X)
            if X.ndim == 3:
                X = np.expand_dims(X, axis=-1)
            assert len(X) == self.num_images
            return X
        except Exception as e:
            if self.verbose:
                print(e)
            return None