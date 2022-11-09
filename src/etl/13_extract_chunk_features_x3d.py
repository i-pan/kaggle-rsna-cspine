import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pandas as pd
import torch

from collections import defaultdict
from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

import sys
sys.path.insert(0, "../../src/")

from skp import builder
from etl.utils import create_dir


def predict(model, X):
    X = torch.from_numpy(X).float().cuda().unsqueeze(0).unsqueeze(0)
    return torch.sigmoid(model(X)).cpu()


CONFIG = "../configs/chunk/chunk000.yaml"
base_name = osp.basename(CONFIG).replace(".yaml", "")
SAVE_FEATURES_DIR = f"../../data/train-{base_name}-features/"


create_dir(SAVE_FEATURES_DIR)
CHUNK_DIR = "../../data/train-numpy-vertebra-chunks/"

torch.set_grad_enabled(False)

IMSIZE = 192.
PLOT = False

df = pd.read_csv("../../data/train_vertebra_chunks_kfold.csv")

cfg = OmegaConf.load(CONFIG)
dataset = builder.build_dataset(cfg, data_info={"inputs": [0], "labels": [0]}, mode="predict")

checkpoints = {
    f"fold{i}": f"../../experiments/{base_name}/sbn/fold{i}/checkpoints/best.ckpt" for i in range(3)
}

models = {}
for fold, ckpt in checkpoints.items():
    cfg.model.load_pretrained = str(ckpt)
    tmp_model = builder.build_model(cfg.copy())
    tmp_model = tmp_model.eval().cuda()
    models[fold] = tmp_model
    create_dir(osp.join(SAVE_FEATURES_DIR, fold))


chunk_df_dict = defaultdict(list)
for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    chunks = np.sort(study_df.filename)
    assert len(chunks) == 7
    chunks = [np.load(osp.join(CHUNK_DIR, c)) for c in chunks]
    for ind, c in enumerate(chunks):
        if len(c) == 0:
            print("FAILED")
            chunks[ind] = np.zeros((1, 1, 1))
    chunks = np.stack([dataset.process_image({"image": dataset.adjust_z_dimension(c)})["image"]
                       for c in chunks])
    chunks = torch.from_numpy(chunks).unsqueeze(1).float().cuda()
    for fold in range(len(models)):
        fold = f"fold{fold}"
        features = torch.cat([models[fold].extract_features(chunks[i].unsqueeze(0))
                              for i in range(chunks.shape[0])]).cpu().numpy()
        np.save(osp.join(SAVE_FEATURES_DIR, fold, study_id + ".npy"), features)
    chunk_df_dict["filename"].append(study_id + ".npy")
    chunk_df_dict["StudyInstanceUID"].append(study_id)

chunk_df = pd.DataFrame(chunk_df_dict)
train_df = pd.read_csv("../../data/train.csv")
chunk_df = chunk_df.merge(train_df, on="StudyInstanceUID")
chunk_df = chunk_df.merge(df[["StudyInstanceUID"] + [c for c in df.columns if "outer" in c or "inner" in c]].drop_duplicates())
chunk_df.to_csv("../../data/train_chunk_features_kfold.csv", index=False)