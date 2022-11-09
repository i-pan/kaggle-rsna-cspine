import cv2
import glob
import numpy as np
import os.path as osp
import pandas as pd

from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from utils import create_dir


IMAGE_SAVEDIR = "../../data/train-numpy-seg-whole-192x192x192"
LABEL_SAVEDIR = "../../data/label-numpy-seg-whole-192x192x192/"
IMSIZE = 192.0

create_dir(IMAGE_SAVEDIR)
create_dir(LABEL_SAVEDIR)

segmentations = glob.glob("../../data/segmentations-numpy/*.npy")
df = pd.read_csv("../../data/train_metadata.csv")

for seg in tqdm(segmentations, total=len(segmentations)):
    study_uid = seg.split("/")[-1].replace(".npy", "")
    tmp_df = df[df.StudyInstanceUID == study_uid].sort_values("ImagePositionPatient_2", ascending=False)
    study_images = tmp_df.filename.tolist()
    study_images = [osp.join("../../data/pngs", i.replace("dcm", "png")) for i in study_images]
    seg_array = np.load(seg)
    assert len(seg_array) == len(study_images)
    img_array = np.stack([cv2.imread(i, 0) for i in study_images])
    assert seg_array.shape == img_array.shape
    z, x, y = img_array.shape
    rescale_factor = [IMSIZE / z, IMSIZE / x, IMSIZE / y]
    seg_array = zoom(seg_array, rescale_factor, order=0, prefilter=False)
    img_array = zoom(img_array, rescale_factor, order=0, prefilter=False)
    assert seg_array.shape == img_array.shape == (IMSIZE, IMSIZE, IMSIZE)
    np.save(osp.join(IMAGE_SAVEDIR, f"{study_uid}.npy"), img_array.astype("uint8"))
    np.save(osp.join(LABEL_SAVEDIR, f"{study_uid}.npy"), seg_array.astype("uint8"))
