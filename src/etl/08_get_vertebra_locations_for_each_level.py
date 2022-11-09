import cv2
import glob
import numpy as np
import os.path as osp
import pandas as pd
import pickle
import torch

from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

import sys
sys.path.insert(0, "../../src/")

from skp import builder


def predict(model, X):
    X = torch.from_numpy(X).float().cuda().unsqueeze(0).unsqueeze(0)
    return torch.sigmoid(model(X)).cpu()


torch.set_grad_enabled(False)

IMSIZE = 192.
CONFIG = "../configs/seg/pseudoseg000.yaml"

df = pd.read_csv("../../data/train_metadata.csv")
cfg = OmegaConf.load(CONFIG)
dataset = builder.build_dataset(cfg, data_info={"inputs": [0], "labels": [0]}, mode="predict")
checkpoints = np.sort(glob.glob(f"../../experiments/{osp.basename(CONFIG).replace('.yaml', '')}/sbn/fold*/checkpoints"
                                f"/best.ckpt"))
models = []
for ckpt in checkpoints:
    cfg.model.load_pretrained = str(ckpt)
    tmp_model = builder.build_model(cfg.copy())
    tmp_model = tmp_model.eval().cuda()
    models.append(tmp_model)

THRESHOLD = 0.4
coords_dict = {}
for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    coords_dict[study_id] = {}
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    study_images = study_df.filename.tolist()
    study_images = [osp.join("../../data/pngs", i.replace("dcm", "png")) for i in study_images]
    X = np.stack([cv2.imread(im, 0) for im in study_images])
    z, h, w = X.shape
    rescale_factor = [IMSIZE / z, IMSIZE / h, IMSIZE / w]
    X = zoom(X, rescale_factor, order=0, prefilter=False)
    X = dataset.preprocess(X)
    p = torch.cat([predict(m, X) for m in models]).mean(0)
    for level in range(7):
        p_level = np.vstack(np.where(p[level].numpy() >= THRESHOLD)).astype("float")
        adjusted_threshold = THRESHOLD
        while p_level.shape[1] == 0 and adjusted_threshold > 0.1:
            print(f"FAILED {study_id} C{level+1} AT THRESHOLD {adjusted_threshold}!")
            adjusted_threshold = round(adjusted_threshold - 0.1, 1)
            p_level = np.vstack(np.where(p[level].numpy() >= adjusted_threshold)).astype("float")
        if p_level.shape[1] == 0:
            print(f"FAILED {study_id} COMPLETELY")
            coords_dict[study_id][f"C{level+1}"] = [0] * 6
            continue
        for axis in range(3):
            p_level[axis] /= rescale_factor[axis]
        p_level = p_level.astype("int")
        coords_dict[study_id][f"C{level+1}"] = [p_level[0].min(), p_level[0].max(),
                                                p_level[1].min(), p_level[1].max(),
                                                p_level[2].min(), p_level[2].max()]


with open("../../data/train_cspine_coords.pkl", "wb") as f:
    pickle.dump(coords_dict, f)

