import pandas as pd

from utils import create_double_cv


df = pd.read_csv("../../data/train.csv")[["StudyInstanceUID"]]

cv_df = create_double_cv(df, "StudyInstanceUID", 5, 5)
cv_df.to_csv("../../data/train_kfold.csv", index=False)