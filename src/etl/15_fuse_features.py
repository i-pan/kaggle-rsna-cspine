import glob
import numpy as np
import os.path as osp

from tqdm import tqdm
from utils import create_dir


SAVE_DIR = "../../data/train-fused-features/"
for fold in range(5):
    create_dir(osp.join(SAVE_DIR, f"fold{fold}"))

features = glob.glob("../../data/train-chunk000-features/fold*/*.npy")

for feat in tqdm(features):
    x = np.load(feat)
    y = np.load(feat.replace("chunk000", "chunk101"))
    z = np.concatenate([x, y], axis=1)
    assert z.shape[0] == 7
    np.save(feat.replace("chunk000", "fused"), z)