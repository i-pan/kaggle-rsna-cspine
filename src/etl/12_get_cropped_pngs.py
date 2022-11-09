import cv2
import numpy as np
import pandas as pd
import os.path as osp
import pickle

from collections import defaultdict
from tqdm import tqdm
from utils import create_dir


def merge_cspine_coords(coords):
    slice_dict = defaultdict(list)
    for level in [f"C{i+1}" for i in range(7)]:
        z1, z2, h1, h2, w1, w2 = coords[level]
        for z_ind in range(z1, z2 + 1, 1):
            slice_dict[z_ind].append([h1, h2, w1, w2])
    for z_ind, ax_coords in slice_dict.items():
        ax_coords = np.stack(ax_coords, axis=0)
        slice_dict[z_ind] = (ax_coords[:, 0].min(), ax_coords[:, 1].max(), ax_coords[:, 2].min(), ax_coords[:, 3].max())
    return slice_dict


SAVE_DIR = "../../data/train-cropped-cspine-pngs/"
create_dir(SAVE_DIR)

with open("../../data/train_cspine_coords.pkl", "rb") as f:
    cspine_coords = pickle.load(f)

df = pd.read_csv("../../data/train_metadata.csv")
cas_df = pd.read_csv("../../data/train_cas_kfold_all.csv")
used_files = []
for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    if study_id not in [*cspine_coords]:
        continue
    study_coords = cspine_coords[study_id]
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    files = study_df.filename.apply(lambda x: x.replace("dcm", "png")).values
    img_array = np.stack([cv2.imread(osp.join("../../data/pngs-with-seg", f)) for f in files])
    Z, H, W = img_array.shape[:3]
    create_dir(osp.join(SAVE_DIR, study_id))
    slice_dict = merge_cspine_coords(study_coords)
    for slice_ind, ax_coords in slice_dict.items():
        h1, h2, w1, w2 = ax_coords
        img_slice = img_array[slice_ind, h1:h2+1, w1:w2+1]
        assert img_slice.ndim == 3
        each_file = files[slice_ind]
        status = cv2.imwrite(osp.join(SAVE_DIR, each_file), img_slice)
        used_files.append(each_file)

cas_df = cas_df[cas_df.filename.isin(used_files)]
cas_df.to_csv("../../data/train_cas_kfold_all_by_level.csv", index=False)


