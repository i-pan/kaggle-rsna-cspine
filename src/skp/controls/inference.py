import os, os.path as osp
import pandas as pd
import pickle

from .. import tasks
from .. import builder 
from .datamaker import get_train_val_datasets, prepend_data_dir
from .training import build_trainer


def predict(cfg, args):
    assert args.inference_checkpoint, "Inference checkpoint not specified"
    assert args.inference_data_dir, "Image directory not provided"
    assert args.inference_imgfiles, "List of images not provided"
    assert args.inference_save, "Save location not provided"

    cfg.model.load_pretrained = args.inference_checkpoint
    cfg.tta = args.inference_tta
    if isinstance(cfg.tta, str):
        assert cfg.tta in ["5crop", "flip"], f"`{cfg.tta}` is not a valid TTA option."
    cfg.inference_act_fn = args.inference_act_fn
    if cfg.tta == "5crop":
        cfg.tta_5crop_size = cfg.transform.crop.params.imsize
        cfg.transform.crop = None

    model = builder.build_model(cfg)
    task = builder.build_task(cfg, model)

    trainer = build_trainer(cfg, args, task)
    twodc = cfg.data.dataset.params.get("channels", None) == "2dc"

    with open(args.inference_imgfiles) as f:
        imgfiles = [prepend_data_dir(line.strip(), args.inference_data_dir, twodc) for line in f.readlines()]

    if args.debug:
        imgfiles = imgfiles[:10]

    data_info = {
        "inputs": imgfiles,
        "labels": [0] * len(imgfiles)
    }

    cfg.data.dataset.params.return_name = True
    dataset = builder.build_dataset(cfg, data_info=data_info, mode="predict")
    loader = builder.build_dataloader(cfg, dataset, "predict")

    predictions = trainer.predict(task, loader)
    
    if not osp.exists(osp.dirname(args.inference_save)):
        os.makedirs(osp.dirname(args.inference_save))

    with open(args.inference_save, "wb") as f:
        pickle.dump(predictions, f)


def test(cfg, args):
    assert args.inference_checkpoint, "Inference checkpoint not specified"
    assert args.inference_save, "Save location not provided"
    assert args.test_kfold > -1, "Must specify fold in test mode"

    cfg.model.load_pretrained = args.inference_checkpoint
    model = builder.build_model(cfg)
    task = builder.build_task(cfg, model)

    trainer = build_trainer(cfg, args, task)

    df = pd.read_csv(cfg.data.annotations)

    # This is actually just the validation dataset
    # Since we are doing single-loop CV, not nested
    cfg.data.outer_fold = args.test_kfold
    cfg.data.dataset.params.return_name = True
    _,test_dataset = get_train_val_datasets(cfg)

    loader = builder.build_dataloader(cfg, test_dataset, "test")

    predictions = trainer.predict(task, loader)
    
    if not osp.exists(osp.dirname(args.inference_save)):
        os.makedirs(osp.dirname(args.inference_save))

    with open(args.inference_save, "wb") as f:
        pickle.dump(predictions, f)
