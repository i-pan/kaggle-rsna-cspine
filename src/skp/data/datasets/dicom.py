from . import helper
from .base import ImageDataset


class XRDICOMDataset(ImageDataset):

    def __init__(self, *args, **kwargs):
        # Default is to convert to 8-bit image
        # However, can also leave in float if more granularity is desired
        # Values will be rescaled to [0, 1], then augmentations+preprocessing will be applied
        # Remember, many augmentations only work with 8-bit images
        self.convert_8bit = kwargs.pop("convert_8bit", True)
        super().__init__(*args, **kwargs)

    def load_dicom(self, dcmfile):
        return helper.load_dicom(dcmfile, mode="XR", convert_8bit=self.convert_8bit,
                                 verbose=self.verbose)

    def get(self, i):
        try:
            dicom = self.load_dicom(self.inputs[i])
            return {"image": dicom}
        except Exception as e:
            if self.verbose:
                print(f'Failed to load {self.inputs[i]} :  {e}')
            return None


class CTDICOMDataset(XRDICOMDataset):

    def __init__(self, *args, **kwargs):
        assert "window" in kwargs, f"You must specify a window or list of windows"
        # If specifying windows, it must be a LIST of LISTS
        # If only 1 window, should be in this format --> [[WL, WW]]
        # Note that `raw_hu` is an option if this is the desired input
        # However, many augmentations only work with uint8, so be aware of this
        self.window = kwargs.pop("window")
        super().__init__(*args, **kwargs)

    def load_dicom(self, dcmfile):
        return helper.load_dicom(dcmfile, mode="CT", convert_8bit=self.convert_8bit,
                                 window=self.window, verbose=self.verbose)


class MRDICOMDataset(XRDICOMDataset):

    def load_dicom(self, dcmfile):
        return helper.load_dicom(dcmfile, mode="MR", convert_8bit=self.convert_8bit,
                                 verbose=self.verbose)