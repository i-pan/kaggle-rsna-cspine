import glob
import numpy as np
import pandas as pd
import pydicom

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class DICOMMetadata(Dataset):

    def __init__(self, dicom_files):
        self.dicom_files = dicom_files

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, i):
        dcm = pydicom.dcmread(self.dicom_files[i], stop_before_pixels=True)
        meta = dict()
        for attrib in ['SOPInstanceUID', 'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'ImagePositionPatient',
                       'ImageOrientationPatient', 'PixelSpacing']:
            meta[attrib] = self.get_dicom_attribute(dcm, attrib)
        meta['filename'] = self.dicom_files[i].split('/')[-1]
        return meta

    @staticmethod
    def get_dicom_attribute(dcm, att):
        try:
            return getattr(dcm, att)
        except Exception as e:
            print(e)
            return None


all_dicoms = glob.glob("../../data/train_images/*/*.dcm")
dataset = DICOMMetadata(all_dicoms)
loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x, num_workers=4)
metadata = []
for each_meta in tqdm(loader):
    metadata.append(each_meta)

metadata = {k: [d[0][k] for d in metadata] for k in metadata[0][0].keys()}
metadata['ImagePositionPatient'] = np.asarray(metadata['ImagePositionPatient'])
metadata['ImagePositionPatient_0'] = metadata['ImagePositionPatient'][:, 0].astype('float32')
metadata['ImagePositionPatient_1'] = metadata['ImagePositionPatient'][:, 1].astype('float32')
metadata['ImagePositionPatient_2'] = metadata['ImagePositionPatient'][:, 2].astype('float32')

del metadata['ImagePositionPatient']
metadata['PixelSpacing'] = np.asarray(metadata['PixelSpacing'])
metadata['PixelSpacing_x'] = metadata['PixelSpacing'][:, 0].astype('float32')
metadata['PixelSpacing_y'] = metadata['PixelSpacing'][:, 1].astype('float32')
del metadata['PixelSpacing']

meta_df = pd.DataFrame(metadata).sort_values(["PatientID", "StudyInstanceUID", "ImagePositionPatient_2"])

# Calculate slice spacing
study_df_list = []
for study_id, study_df in meta_df.groupby("StudyInstanceUID"):
    z_position = np.sort(study_df.ImagePositionPatient_2)
    spacings = z_position[:-1] - z_position[1:]
    study_df["SliceSpacing"] = np.median(np.abs(spacings))
    study_df_list.append(study_df)

meta_df = pd.concat(study_df_list)
meta_df["filename"] = meta_df["StudyInstanceUID"] + "/" + meta_df["filename"]
meta_df.to_csv("../../data/train_metadata.csv", index=False)
