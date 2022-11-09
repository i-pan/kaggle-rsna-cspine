import glob
import pandas as pd


studies = glob.glob("../../data/train-numpy-seg-whole-192x192x192/*")
labels = [s.replace("train", "label") for s in studies]

df = pd.DataFrame({"filename": studies, "label": labels})
df["filename"] = df.filename.apply(lambda x: x.replace("../../data/", ""))
df["label"] = df.label.apply(lambda x: x.replace("../../data/", ""))

df["StudyInstanceUID"] = df.filename.apply(lambda x: x.split("/")[-1].replace(".npy", ""))

folds_df = pd.read_csv("../../data/train_kfold.csv")
df = df.merge(folds_df, on="StudyInstanceUID")

df.to_csv("../../data/train_seg_whole_192_kfold.csv", index=False)