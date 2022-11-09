import albumentations as A
import cv2
import numpy as np

from scipy.ndimage.interpolation import zoom
from .transforms import compose


class Resize3D:

    def __init__(self, imsize, order=1):
        self.imsize = imsize
        self.order = order

    def __call__(self, x):
        # Assumes channels-LAST
        scale_factor = [self.imsize[0] / x.shape[0],
                        self.imsize[1] / x.shape[1],
                        self.imsize[2] / x.shape[2]]
        if x.ndim == 4:
            scale_factor.append(1)
        return zoom(x, scale_factor, order=self.order, prefilter=self.order > 1)


class Crop3D:

    def __init__(self, imsize, mode="random"):
        assert mode in ["random", "center"]
        self.imsize = imsize
        self.mode = mode

    def __call__(self, data):
        z_diff = data["image"].shape[0] - self.imsize[0]
        h_diff = data["image"].shape[1] - self.imsize[1]
        w_diff = data["image"].shape[2] - self.imsize[2]
        if self.mode == "center":
            z_min = z_diff // 2
            h_min = h_diff // 2
            w_min = w_diff // 2
        elif self.mode == "random":
            z_min = np.random.randint(0, z_diff + 1) if z_diff > 0 else 0
            h_min = np.random.randint(0, h_diff + 1) if h_diff > 0 else 0
            w_min = np.random.randint(0, w_diff + 1) if w_diff > 0 else 0
        z_max = z_min + self.imsize[0]
        h_max = h_min + self.imsize[1]
        w_max = w_min + self.imsize[2]
        for k, v in data.items():
            data[k] = v[z_min:z_max, h_min:h_max, w_min:w_max]
        assert data["image"].shape[:3] == self.imsize, f"Cropped shape {data['image'].shape[:3]} " \
                                                       f"does not match specified shape {self.imsize}"
        return data


# Preserves aspect ratio
# Pads the rest
def resize_3d(imsize):
    x, y = imsize
    return compose([
        A.LongestMaxSize(max_size=max(x, y), p=1),
        A.PadIfNeeded(min_height=x, min_width=y, p=1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
    ], num_images=256, p=1)


# This can be more useful if training w/ crops and rectangular images
def resize_alt_3d(imsize):
    x, y = imsize
    return compose([
        A.SmallestMaxSize(max_size=max(x, y), p=1)
    ], num_images=256, p=1)


# Ignore aspect ratio
def resize_ignore_3d(imsize, order=1):
    # x, y = imsize
    # return compose([
    #     A.Resize(x, y, p=1)
    # ], num_images=256, p=1)
    return Resize3D(imsize=imsize, order=order)


def crop_3d(imsize, mode):
    # x, y = imsize
    # if mode == 'train':
    #     cropper = A.RandomCrop(height=x, width=y, p=1)
    # else:
    #     cropper = A.CenterCrop(height=x, width=y, p=1)
    # return compose([
    #     cropper
    # ], num_images=256, p=1)
    if mode == "train":
        cropper = Crop3D(imsize=imsize, mode="random")
    else:
        cropper = Crop3D(imsize=imsize, mode="center")
    return cropper


def rand_augment_3d(p, n, spatial_only=False, dropout=False):
    """
    Similar to RandAugment strategy.

    :param p: probability of applying transform; note this is the probability of applying ANY transform
              e.g., if p=0.1, then expect 10% of images to be NON-augmented; p=1 for original RA
    :param n: number of transforms to apply to an image
    :param spatial_only: only include transforms that can be used for any array type, not just 8-bit
    :param dropout: include CoarseDropout
    :param num_images: number of images to augment (used for 3D data)
    :return: albumentations Compose transform
    """
    augmentations = [
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.0, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.0, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.GridDistortion(p=1),
        A.GaussianBlur(p=1),
        A.GaussNoise(p=1)
    ]
    if not spatial_only:
        augmentations += [
            A.RandomGamma(p=1),
            A.RandomBrightnessContrast(contrast_limit=0.2, brightness_limit=0.0, p=1),
            A.RandomBrightnessContrast(contrast_limit=0.0, brightness_limit=0.2, p=1),
        ]
    if dropout:
        augmentations += [A.CoarseDropout(min_height=0.05, max_height=0.2,
                                          min_width=0.05, max_width=0.2,
                                          min_holes=2, max_holes=8, fill_value=0, p=1)]
    return compose(A.SomeOf(augmentations, n=n, p=1, replace=False), num_images=256, p=p)
