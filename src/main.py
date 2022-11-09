import argparse
import os, os.path as osp
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf

from skp.controls import datamaker, training, inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str)
    parser.add_argument("config", type=str)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--kfold", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)

    parser.add_argument("--inference-checkpoint", type=str)
    parser.add_argument("--inference-data-dir", type=str)
    parser.add_argument("--inference-imgfiles", type=str)
    parser.add_argument("--inference-act-fn", type=str, default=None)
    parser.add_argument("--inference-tta", type=str, default=None)
    parser.add_argument("--inference-cpu", type=str)
    parser.add_argument("--inference-save", type=str)
    parser.add_argument("--inference-cam-class", type=int)

    parser.add_argument("--offline", action="store_true", help="run wandb in offline mode")
    parser.add_argument("--group-by-seed", action="store_true", help="group models by seed")

    parser.add_argument("--debug", action="store_true")

    parser = pl.Trainer.add_argparse_args(parser)

    return parser.parse_args()


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def read_config(cfgfile):
    cfg = OmegaConf.load(cfgfile)
    while "base" in cfg:
        base_cfg = cfg.pop("base")
        cfg = update(OmegaConf.load(base_cfg), cfg)
    return cfg


def main(args):
    # Print some info ...
    print('PyTorch environment ...')
    print(f'  torch.__version__              = {torch.__version__}')
    print(f'  torch.version.cuda             = {torch.version.cuda}')
    print(f'  torch.backends.cudnn.version() = {torch.backends.cudnn.version()}')
    print('\n')

    # Load config
    cfg = read_config(args.config)

    # Set seed, if specified in command line
    if args.seed >= 0:
        cfg.experiment.seed = args.seed

    # Set strategy in config using command line
    cfg.strategy = args.strategy

    # Handle experiment name
    cfg.experiment.project = osp.basename(osp.dirname(os.getcwd()))
    cfg.experiment.name = osp.basename(args.config).replace('.yaml', '')

    print(f"=== PROJECT <{cfg.experiment.project}> ===")
    print(f'Running experiment {cfg.experiment.name} ...')

    # If using SyncBatchNorm, create subfolder
    if args.sync_batchnorm:
        cfg.experiment.name = osp.join(cfg.experiment.name, 'sbn')

    # If running K-fold, change seed and save directory
    # Also need to edit folds in data
    if args.kfold >= 0:
        cfg.experiment.seed = int(f'{cfg.experiment.seed}{args.kfold}')
        if isinstance(cfg.data.get("inner_fold"), (int, float)):
            print("Command line K-fold training only supports single outer loop ...")
            print(f"Inner fold currently set to {cfg.data.inner_fold} ...")
            print("Setting to `None` ...")
            cfg.data.inner_fold = None
        if cfg.data.outer_fold != args.kfold:
            print(f"Outer fold currently set to {cfg.data.outer_fold} ...")
            print(f"Changing to {args.kfold} ...")
        cfg.data.outer_fold = args.kfold

    if args.mode != "train":
        cfg.experiment.save_dir = osp.join(cfg.experiment.save_dir, args.mode)

    print(f'Saving checkpoints and logs to {cfg.experiment.save_dir} ...')

    # Set number of workers
    if cfg.data:
        cfg.data.num_workers = args.num_workers

        # Set seed
    assert hasattr(cfg.experiment, 'seed'), \
        'Please specify `seed` under `experiment` in config file or in command line'
    seed = pl.seed_everything(cfg.experiment.seed)

    if args.mode in ["train"]:
        getattr(training, args.mode)(cfg, args)
    else:
        getattr(inference, args.mode)(cfg, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)