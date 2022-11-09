import pandas as pd
import os.path as osp

from ..builder import build_dataset


def get_train_val_test_splits(cfg, df):
    if 'split' in df.columns and cfg.data.use_fixed_splits:
        train_df = df[df.split == 'train']
        valid_df = df[df.split == 'valid']
        test_df  = df[df.split == 'test']
        return train_df, valid_df, test_df

    i, o = cfg.data.get("inner_fold"), cfg.data.outer_fold
    if isinstance(i, (int,float)):
        print(f'<inner fold> : {i}')
        print(f'<outer fold> : {o}')
        test_df = df[df.outer == o]
        df = df[df.outer != o]
        train_df = df[df[f'inner{o}'] != i]
        valid_df = df[df[f'inner{o}'] == i]
        valid_df = valid_df.drop_duplicates().reset_index(drop=True)
        test_df = test_df.drop_duplicates().reset_index(drop=True)
    else:
        print('No inner fold specified ...')
        print(f'<outer fold> : {o}')
        train_df = df[df.outer != o]
        valid_df = df[df.outer == o]
        valid_df = valid_df.drop_duplicates().reset_index(drop=True)
        test_df = valid_df.copy()

    return train_df, valid_df, test_df


def prepend_data_dir(fp, data_dir, twodc=False):
    if twodc:
        # fp is a string with filepaths separated by commas
        # Need to split into component filepaths,
        # prepend data directory,
        # then rejoin. 
        fp = fp.split(",")
        fp = [osp.join(data_dir, _) for _ in fp]
        fp = ",".join(fp)
    else:
        fp = osp.join(data_dir, fp)
    return fp


def get_train_val_datasets(cfg):
    INPUT_COL = cfg.data.input
    LABEL_COL = cfg.data.target

    df = pd.read_csv(cfg.data.annotations)

    train_df, valid_df, _ = get_train_val_test_splits(cfg, df)

    data_dir = cfg.data.data_dir
    if "foldx" in data_dir:
        data_dir = data_dir.replace("foldx", f"fold{cfg.data.outer_fold}")

    twodc = cfg.data.dataset.params.get("channels", None) == "2dc" or \
            cfg.data.dataset.params.get("load_mode", None) == "list"

    train_inputs = [prepend_data_dir(_, data_dir, twodc) for _ in train_df[INPUT_COL]]
    valid_inputs = [prepend_data_dir(_, data_dir, twodc) for _ in valid_df[INPUT_COL]]

    train_labels = train_df[LABEL_COL].values
    valid_labels = valid_df[LABEL_COL].values

    if cfg.data.dataset.params.get("segmentation_format", None) in ["png","jpg","jpeg","npy","numpy"]:
        train_labels = [osp.join(data_dir, _) for _ in train_labels]
        valid_labels = [osp.join(data_dir, _) for _ in valid_labels]
    
    if cfg.data.dataset.params.get("segmentation_format", None) == "multislice_pred":
        train_labels = [prepend_data_dir(_, data_dir, twodc) for _ in train_df[LABEL_COL]]
        valid_labels = [prepend_data_dir(_, data_dir, twodc) for _ in valid_df[LABEL_COL]]

    train_data_info = dict(inputs=train_inputs, labels=train_labels)
    valid_data_info = dict(inputs=valid_inputs, labels=valid_labels)

    train_dataset = build_dataset(cfg, 
        data_info=train_data_info,
        mode='train')
    valid_dataset = build_dataset(cfg, 
        data_info=valid_data_info,
        mode='valid')

    print(f'TRAIN : n={len(train_dataset)}')
    print(f'VALID : n={len(valid_dataset)}')

    return train_dataset, valid_dataset


