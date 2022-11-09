import glob
import pandas as pd


df = pd.read_csv("../../data/train_seg_whole_192_kfold.csv")
pseudosegs = glob.glob("../../data/train-pseudo-seg-numpy/*")

fold_cols = [c for c in df.columns if "inner" in c or "outer" in c]
pseudo_df = pd.DataFrame({"filename": pseudosegs})
pseudo_df["filename"] = pseudo_df.filename.apply(lambda x: x.replace("../../data/", ""))
pseudo_df["label"] = pseudo_df.filename.apply(lambda x: x.replace("train", "label"))
pseudo_df[fold_cols] = -1
df = pd.concat([df, pseudo_df])
df.to_csv("../../data/train_seg_whole_192_kfold_with_pseudo.csv", index=False)