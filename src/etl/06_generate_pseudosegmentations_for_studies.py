import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pandas as pd
import torch

from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

import sys
sys.path.insert(0, "../../src/")

from skp import builder
from utils import create_dir


def predict(model, X):
    X = torch.from_numpy(X).float().cuda().unsqueeze(0).unsqueeze(0)
    return torch.sigmoid(model(X)).cpu()


torch.set_grad_enabled(False)

IMAGE_SAVEDIR = "../../data/train-pseudo-seg-numpy"
LABEL_SAVEDIR = "../../data/label-pseudo-seg-numpy"
IMSIZE = 192.
PLOT = False

create_dir(IMAGE_SAVEDIR)
create_dir(LABEL_SAVEDIR)

df = pd.read_csv("../../data/train_metadata.csv")
existing_segmentations = glob.glob("../../data/segmentations/*.nii")
df = df[~df.StudyInstanceUID.isin([es.split("/")[-1].replace(".nii", "") for es in existing_segmentations])]

cfg = OmegaConf.load("../configs/seg/seg000.yaml")
dataset = builder.build_dataset(cfg, data_info={"inputs": [0], "labels": [0]}, mode="predict")
checkpoints = np.sort(glob.glob("../../experiments/seg000/sbn/fold*/checkpoints/last.ckpt"))
models = []
for ckpt in checkpoints:
    cfg.model.load_pretrained = str(ckpt)
    tmp_model = builder.build_model(cfg.copy())
    tmp_model = tmp_model.eval().cuda()
    models.append(tmp_model)

for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    study_images = study_df.filename.tolist()
    study_images = [osp.join("../../data/pngs", i.replace("dcm", "png")) for i in study_images]
    X = np.stack([cv2.imread(im, 0) for im in study_images])
    z, h, w = X.shape
    rescale_factor = [IMSIZE / z, IMSIZE / h, IMSIZE / w]
    X = zoom(X, rescale_factor, order=0, prefilter=False)
    np.save(osp.join(IMAGE_SAVEDIR, f"{study_id}.npy"), X.astype("uint8"))
    X = dataset.preprocess(X)
    p = torch.cat([predict(m, X) for m in models]).mean(0)
    p_spine = p.sum(0)
    p = torch.argmax(p, dim=0) + 1
    p[p_spine < 0.5] = 0
    p = p.numpy()
    p = p.astype("uint8")
    np.save(osp.join(LABEL_SAVEDIR, f"{study_id}.npy"), p)
    if PLOT:
        p_clone = p.copy().astype("float32")
        p_clone /= 8.
        p_clone *= 255.
        for i in range(0, p_clone.shape[-1], 10):
            if np.sum(p_clone[..., i]) > 0:
                plt.imshow(p_clone[:,:,i], cmap="gray")
                plt.show()

