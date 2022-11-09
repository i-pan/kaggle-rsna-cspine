import glob
import nibabel
import numpy as np
import os.path as osp

from tqdm import tqdm
from utils import create_dir


SAVEDIR = "../../data/segmentations-numpy/"
create_dir(SAVEDIR)

segmentations = glob.glob("../../data/segmentations/*.nii")
print(f"{len(segmentations)} segmentations are available !")

labels = []
for seg in tqdm(segmentations, total=len(segmentations)):
    seg_array = nibabel.load(seg).get_fdata()[:, ::-1, ::-1].transpose(2, 1, 0)
    seg_array[seg_array > 7] = 8
    np.save(osp.join(SAVEDIR, seg.split("/")[-1].replace("nii", "npy")), seg_array.astype("uint8"))
    labels.extend(list(np.unique(seg_array)))

print(f"CLASSES : {np.unique(labels)}")

