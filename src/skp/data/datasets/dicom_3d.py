import glob
import numpy as np
import os.path as osp

from . import helper
from .base_3d import ImageStackDataset


class CTDICOMStackDataset(ImageStackDataset):

    def __init__(self, *args, **kwargs):
        assert "window" in kwargs, f"You must specify a window or list of windows"
        # If specifying windows, it must be a LIST of LISTS
        # If only 1 window, should be in this format --> [[WL, WW]]
        # Note that `raw_hu` is an option if this is the desired input
        # However, many augmentations only work with uint8, so be aware of this
        self.convert_8bit = kwargs.pop("convert_8bit", True)
        self.window = kwargs.pop("window")
        self.orientation = kwargs.pop("orientation", "axial")
        self.orientation_dict = {"coronal": 0, "sagittal": 1, "axial": 2}
        super().__init__(*args, **kwargs)

    def load_dicom(self, dcmfile):
        try:
            return helper.load_dicom(dcmfile, mode="CT", convert_8bit=self.convert_8bit,
                                     window=self.window, return_position=True, verbose=self.verbose)
        except Exception as e:
            if self.verbose:
                print(e)
            return None

    def get(self, i):
        try:
            # This assumes that the list of inputs given contains
            # DICOM directories and that all files within the directory
            # are valid image DICOMs
            #
            # TODO: implement other variations, such as when the input
            # is a list of dicom files or a DataFrame containing
            # the DICOM file paths
            dicoms = np.sort(glob.glob(osp.join(self.inputs[i], '*')))
            dicoms = [self.load_dicom(fp) for fp in dicoms]
            dicoms = [d for d in dicoms if not isinstance(d, type(None))]
            if len(dicoms) == 0:
                print(f"No DICOMs found for {self.inputs[i]}")
                return None
            array, positions = [d[0] for d in dicoms], [d[1] for d in dicoms]
            positions = np.asarray(positions)
            # positions.shape = (num_images, 3)
            assert positions.shape[1] == 3 and positions.ndim == 2
            positions = positions[:, self.orientation_dict[self.orientation]]
            positions = np.argsort(positions)
            array = np.concatenate(array, axis=0)
            array = array[positions]
            assert len(array) == self.num_images
            return {"image": array}
        except Exception as e:
            if self.verbose:
                print(f'Failed to load {self.inputs[i]} :  {e}')
            return None


class MRDICOMStackDataset(CTDICOMStackDataset):

    def load_dicom(self, dcmfile):
        try:
            return helper.load_dicom(dcmfile, mode="MR", convert_8bit=self.convert_8bit,
                                     window=None, return_position=True, verbose=self.verbose)
        except Exception as e:
            if self.verbose:
                print(e)
            return None
