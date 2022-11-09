import glob
import numpy as np
import pandas as pd


bbox_df = pd.read_csv("../../data/train_bounding_boxes.csv")
folds_df = pd.read_csv("../../data/train_kfold.csv")

bbox_df["study_slice"] = bbox_df.StudyInstanceUID + "_" + bbox_df.slice_number.astype("str")
cas_dfs = np.sort(glob.glob("../../data/train_cas_fold*.csv"))
cas_dfs = [pd.read_csv(_) for _ in cas_dfs]
for each_cas_df in cas_dfs[1:]:
    cas_dfs[0]["cas"] += each_cas_df["cas"]

cas_df = cas_dfs[0].copy()
del cas_dfs
cas_df["slice_number"] = cas_df.filename.apply(lambda x: x.split("/")[-1].split(".")[0])
cas_df["study_slice"] = cas_df.StudyInstanceUID + "_" + cas_df.slice_number

cas_df["fracture_cas"] = 0
cas_df.loc[cas_df.study_slice.isin(bbox_df.study_slice.tolist()), "fracture_cas"] = 1
cas_df.loc[cas_df.cas >= 0.5, "fracture_cas"] = 1
cas_df["cspine_present"] = cas_df[[f"C{i+1}_present" for i in range(7)]].sum(1)
cas_df = cas_df[cas_df.cspine_present > 0]
print(cas_df.fracture_cas.value_counts(normalize=True))
cas_df = cas_df.merge(folds_df, on="StudyInstanceUID")
cas_df["filename"] = cas_df.filename.apply(lambda x: x.replace("dcm", "png"))
cas_df.to_csv("../../data/train_cas_kfold_all.csv", index=False)