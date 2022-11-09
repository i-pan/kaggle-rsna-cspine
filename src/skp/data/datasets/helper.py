import numpy as np
import pydicom
import random

from pydicom.pixel_data_handlers.util import apply_voi_lut


def coinflip():
    return random.random() > 0.5


def flip_array_2d(data):
    if coinflip():
        for k, v in data.items():
            if k in ["image", "mask"]:
                data[k] = v[::-1, :]
    if coinflip():
        for k, v in data.items():
            if k in ["image", "mask"]:
                data[k] = v[:, ::-1]
    return data


def flip_array_3d(data):
    data = flip_array_2d(data)
    if coinflip():
        for k, v in data.items():
            if k in ["image", "mask"]:
                data[k] = v[:, :, ::-1]
    return data


def transpose_2d(data):
    if coinflip():
        for k, v in data.items():
            if k in ["image", "mask"]:
                assert v.shape[0] == v.shape[1]
                data[k] = v.swapaxes(0, 1)
    return data


def transpose_3d(data):
    # If Z != H != W, no transpose operation can be applied
    img_shape = data["image"].shape
    assert len(np.unique(img_shape[:3])) < 3
    if coinflip() and np.all(img_shape == img_shape[0]):
        # If volume is a cube, can rearrange the axes in any permutation
        axes = np.asarray([0, 1, 2])
        np.random.shuffle(axes)
        for k, v in data.items():
            if k in ["image", "mask"]:
                if v.ndim == 4:
                    data[k] = v.transpose(axes[0], axes[1], axes[2], 3)
                elif v.ndim == 3:
                    data[k] = v.transpose(axes[0], axes[1], axes[2])
                else:
                    raise Exception(f"{k} does not have valid dimensions")
    else:
        # If volume is not a cube, can only rearrange axes which are the same size
        # Most commonly, the axial dimensions will be the same size
        transposed = False
        if coinflip() and img_shape[1] == img_shape[2] and not transposed:
            # Z,H,W -> Z,W,H
            for k, v in data.items():
                data[k] = v.swapaxes(1, 2)
            transposed = True
        if coinflip() and img_shape[0] == img_shape[2] and not transposed:
            # Z,H,W -> W,H,Z
            for k, v in data.items():
                data[k] = v.swapaxes(0, 2)
            transposed = True
        if coinflip() and img_shape[0] == img_shape[1] and not transposed:
            # Z,H,W -> W,H,Z
            for k, v in data.items():
                data[k] = v.swapaxes(0, 1)
    return data


def invert(data):
    if coinflip():
        data["image"] = np.invert(data["image"])
    return data


def reverse_channels(data):
    # Assumes channels-LAST
    assert data["image"].shape[-1] > 1
    if coinflip():
        data["image"] = data["image"][..., ::-1]
    return data


def apply_window(array, window):
    WL, WW = window
    WL, WW = float(WL), float(WW)
    lower, upper = WL - WW / 2, WL + WW / 2
    array = np.clip(array, lower, upper) 
    array = array - lower
    array = array / (upper - lower)
    return array 


def load_dicom(dcmfile, mode, convert_8bit=True, window=None, return_position=False,
               verbose=True):
    assert mode.lower() in ["xr", "ct", "mr"], f"{mode} is not a valid argument for `mode`"

    dicom = pydicom.dcmread(dcmfile)

    if return_position:
        assert hasattr(dicom, "ImagePositionPatient"), "DICOM metadata does not have ImagePositionPatient attribute"

    if mode == "xr": 
        # Apply the lookup table, if possible
        try:
            array = apply_voi_lut(dicom.pixel_array, dicom)
        except Exception as e:
            if verbose: print(e)
        # Rescale to [0, 1] using min and max values 
        array = array.astype("float32") 
        # Invert image, if needed
        if hasattr(dicom, "PhotometricInterpretation"):
            if dicom.PhotometricInterpretation == "MONOCHROME1":
                array = np.amax(array) - array
        else:
            if verbose:
                print(f"{dcmfile} does not have attribute `PhotometricInterpretation`")
        array = array - np.min(array) 
        array = array / np.max(array)
        if convert_8bit:
            array = (array * 255.0).astype("uint8")

        array = np.expand_dims(array, axis=-1)

    elif mode == "ct": 
        # No need to apply LUT, we will assume values are HU
        # Rescale using intercept and slope
        M = float(dicom.RescaleIntercept)
        B = float(dicom.RescaleSlope)
        array = dicom.pixel_array.astype("float32")
        array = M*array + B
        # Apply window, if desired
        if isinstance(window, str):
            assert window == "raw_hu"
            array = np.expand_dims(array, axis=-1) 
        elif not isinstance(window, type(None)):
            array_list = []
            for each_window in window:
                array_list += [
                    np.expand_dims(apply_window(dicom, each_window), axis=-1)
                ]
            if convert_8bit:
                array_list = [(a * 255.0).astype("uint8") for a in array_list]
            # Each window is a dimension in the channel (last) axis
            array = np.concatenate(array_list, axis=-1) 
            # Sanity check
            assert array.shape[2] == len(window)
        else:
            raise Exception("You must provide window(s) or specify `raw_hu`")

    elif mode == "mr": 
        # Apply the lookup table, if possible
        try:
            array = apply_voi_lut(dicom.pixel_array, dicom)
        except Exception as e:
            if verbose: print(e)
        # Rescale to [0, 1] using min and max values 
        # Clip values to be within 2nd and 98th percentile
        array = array.astype("float32") 
        array = np.clip(array, np.percentile(array, 2), np.percentile(array, 98))
        array = array - np.min(array) 
        array = array / np.max(array)
        if convert_8bit:
            array = (array * 255.0).astype("uint8")

        array = np.expand_dims(array, axis=-1)

    if return_position: 
        return array, [float(i) for i in dicom.ImagePositionPatient]
        
    return array