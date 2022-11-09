import cv2
import numpy as np
import os
import os.path as osp
import pandas as pd
import pydicom

from tqdm import tqdm


def create_dir(d):
    if not osp.exists(d):
        os.makedirs(d)


def window(x, WL=40, WW=80):
    # Default is standard brain window
    upper, lower = WL+WW//2, WL-WW//2
    x = np.clip(x, lower, upper)
    x = x - lower
    x = x / (upper - lower)
    x = x * 255
    x = x.astype('uint8')
    return x


def get_array_from_dicom(dcm):
    array = dcm.pixel_array.astype('float32')
    array = array * float(dcm.RescaleSlope)
    array = array + float(dcm.RescaleIntercept)
    return array


SAVEDIR = "../../data/pngs"

create_dir(SAVEDIR)

meta_df = pd.read_csv("../../data/train_metadata.csv")

for row_num, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
    save_dir = osp.join(SAVEDIR, row.StudyInstanceUID)
    save_file = osp.join(save_dir, row.filename.split("/")[-1].replace("dcm", "png"))
    if osp.exists(save_file):
        continue
    dicom = pydicom.dcmread(osp.join("../../data/train_images", row.filename))
    img = get_array_from_dicom(dicom)
    img = window(img, WL=400, WW=2500)
    create_dir(save_dir)
    status = cv2.imwrite(save_file, img)


