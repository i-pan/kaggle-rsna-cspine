import cv2
import numpy as np
import os.path as osp
import pandas as pd
import pickle
import torch

from collections import defaultdict
from omegaconf import OmegaConf
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

import sys
sys.path.insert(0, "../../src/")

from skp import builder


def get_cam(model, x, labels=["fracture"]):
    with torch.no_grad():
        x = model.backbone(x)
        x = x.squeeze(0)
        linear_weights = model.classifier.weight
        cam_dict = {}
        for class_idx, class_w in enumerate(linear_weights):
            cam_dict[labels[class_idx]] = torch.zeros((x.size(-3), x.size(-2), x.size(-1))).to(x.device)
            for idx, w in enumerate(class_w):
                cam_dict[labels[class_idx]] += w * x[idx]
    return {k: v.cpu().numpy() for k, v in cam_dict.items()}


torch.set_grad_enabled(False)

with open("../../data/train_cspine_coords.pkl", "rb") as f:
    cspine_coords = pickle.load(f)


cfg = OmegaConf.load("../configs/chunk/chunk000.yaml")
dataset = builder.build_dataset(cfg, data_info={"inputs": [0], "labels": [0]}, mode="predict")

checkpoints = {
    f"fold{i}": f"../../experiments/chunk000/sbn/fold{i}/checkpoints/best.ckpt" for i in range(5)
}

df = pd.read_csv("../../data/train_metadata_with_cspine_level.csv")
train_df = pd.read_csv("../../data/train.csv")
# Create dictionary mapping study ID to fractured vertebra for easier access
train_dict = {}
for rownum, row in train_df.iterrows():
    train_dict[row.StudyInstanceUID] = row.index[row == 1].tolist()

models = {}
for fold, ckpt in checkpoints.items():
    cfg.model.load_pretrained = str(ckpt)
    tmp_model = builder.build_model(cfg.copy())
    tmp_model = tmp_model.eval().cuda()
    models[fold] = tmp_model

new_df_list = defaultdict(list)

for study_id, study_df in tqdm(df.groupby("StudyInstanceUID"), total=len(df.StudyInstanceUID.unique())):
    # Skip negatives
    if len(train_dict[study_id]) == 0:
        study_df["cas"] = 0
        for fold in range(5):
            new_df_list[f"fold{fold}"].append(study_df)
        continue
    # Sort
    study_df = study_df.sort_values("ImagePositionPatient_2", ascending=False)
    files = study_df.filename.apply(lambda x: x.replace("dcm", "png")).tolist()
    # Load in 3D array
    img_array = np.stack([cv2.imread(osp.join("../data/pngs", f), 0) for f in files])
    Z, H, W = img_array.shape

    for fold, mod in models.items():
        cas_for_study, num_vals_per_slice = np.zeros((len(study_df),)), np.zeros((len(study_df),))
        for level in [f"C{i+1}" for i in range(7)]:
            # Skip non-fractured vertebra
            if level not in train_dict[study_id]:
                continue
            z_axis = np.where(study_df[f"{level}_present"].values == 1)[0]
            # Extract chunk while adding buffer
            z1, z2, h1, h2, w1, w2 = cspine_coords[study_id][level]
            img_chunk = img_array[z1:z2+1, h1:h2+1, w1:w2+1]
            assert len(z_axis) == len(img_chunk)
            # Preprocess chunk
            img_chunk = dataset.process_image({"image": np.expand_dims(dataset.adjust_z_dimension(img_chunk), axis=-1)})
            img_chunk = torch.from_numpy(img_chunk["image"]).unsqueeze(0).float().cuda()
            # Inference on chunk
            cam = get_cam(mod, img_chunk)["fracture"]
            # Reshape and take the max to get the CAS
            cas = cam.reshape(cam.shape[0], -1).max(-1)
            # Scale to [0, 1]
            cas[cas < 0] = 0
            cas /= np.max(cas)
            cas = zoom(cas, (z2 - z1) / float(len(cas)), order=0, prefilter=False)
            cas_for_study[z1:z2] += cas
            num_vals_per_slice[z1:z2] += 1.0
        # Adjust CAS
        cas_for_study[cas_for_study != 0] /= num_vals_per_slice[cas_for_study != 0]
        study_df["cas"] = cas_for_study
        new_df_list[fold].append(study_df)


for fold, each_df_list in new_df_list.items():
    each_df = pd.concat(each_df_list)
    each_df.to_csv(f"../../data/train_cas_{fold}.csv", index=False)






