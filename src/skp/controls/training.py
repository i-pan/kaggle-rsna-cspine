import numpy as np
import os, os.path as osp
import pytorch_lightning as pl
import re
import torch

from omegaconf import ListConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.layer_sync import NativeSyncBatchNorm
from timm.models.layers.norm_act import convert_sync_batchnorm

from torch.nn import Module

from .. import builder
from .. import metrics
from .datamaker import get_train_val_datasets


class TimmSyncBatchNorm(NativeSyncBatchNorm):
    """
    Default SyncBN plugin for Lightning does not work with latest version of timm
    EfficientNets because it uses the native PyTorch `convert_sync_batchnorm` function.

    Use this plugin instead, which uses the timm helper and should work for non-timm
    models as well.
    """
    def apply(self, model: Module) -> Module:
        return convert_sync_batchnorm(model)


def build_elements(cfg):
    # Create model
    model = builder.build_model(cfg)
    # Create loss
    criterion = builder.build_loss(cfg)
    # Create optimizer
    optimizer = builder.build_optimizer(cfg, model.parameters())
    # Create learning rate scheduler
    scheduler = builder.build_scheduler(cfg, optimizer)
    return model, criterion, optimizer, scheduler


def build_trainer(cfg, args, task, snapshot=-1):
    version = f"snapshot_{snapshot}" if snapshot >= 0 else ""

    if isinstance(cfg.data.get("inner_fold"), (int, float)):
        fold_subfolder = f"i{int(cfg.data.inner_fold)}o{int(cfg.data.outer_fold)}"
    else:
        fold_subfolder = f"fold{int(cfg.data.outer_fold)}"

    if args.group_by_seed:
        fold_subfolder = f"{fold_subfolder}_seed{cfg.experiment.seed}"

    save_experiment_path = osp.join(cfg.experiment.save_dir, 
        cfg.experiment.name, 
        fold_subfolder, 
        version)

    if not osp.exists(save_experiment_path):
        os.makedirs(save_experiment_path)

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            # Set dirpath explicitly to save checkpoints in the desired folder
            # This is so that we can keep the desired directory structure and format locally
            # while still using W&B
            dirpath=osp.join(save_experiment_path, "checkpoints"),
            monitor="vm",
            filename="{epoch:03d}-{vm:.4f}",
            save_last=True,
            save_weights_only=True,
            mode=cfg.evaluate.mode,
            save_top_k=cfg.evaluate.get("save_top_k") or 1,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    if cfg.train.get("early_stopping"):
        print(">> Using early stopping ...")
        early_stopping = pl.callbacks.EarlyStopping(
            patience=cfg.train.early_stopping.patience,
            monitor="vm",
            min_delta=cfg.train.early_stopping.min_delta,
            verbose=cfg.train.early_stopping.verbose or False,
            mode=cfg.evaluate.mode,
        )
        callbacks.append(early_stopping)

    if cfg.strategy == "ddp": 
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)
        plugins = [TimmSyncBatchNorm()]
    else:
        strategy = cfg.strategy
        plugins = None

    group_name = args.mode + "_" if args.mode != "train" else "" + cfg.experiment.name.replace("/", "_")
    wandb_logger = WandbLogger(project=cfg.experiment.project,
                               group=group_name,
                               job_type=fold_subfolder,
                               save_dir=save_experiment_path,
                               offline=args.offline)
    
    trainer = pl.Trainer.from_argparse_args(
        args,
        max_epochs=cfg.train.num_epochs,
        callbacks=callbacks,
        plugins=plugins,
        logger=wandb_logger,
        strategy=strategy,
        replace_sampler_ddp=False,
        accumulate_grad_batches=cfg.train.get("accumulate_grad_batches") or 1,
        profiler="simple",
    )

    return trainer


def define_task(cfg, args):
    train_dataset, valid_dataset = get_train_val_datasets(cfg)
    model, loss_fn, optimizer, scheduler = build_elements(cfg)
    evaluation_metrics = [getattr(metrics, m)() for m in cfg.evaluate.metrics]
    valid_metric = (
        list(cfg.evaluate.monitor)
        if isinstance(cfg.evaluate.monitor, ListConfig)
        else cfg.evaluate.monitor
    )

    task = builder.build_task(cfg, model)

    task.set("optimizer", optimizer)
    task.set("scheduler", scheduler)
    task.set("loss_fn", loss_fn)
    task.set("metrics", evaluation_metrics)
    task.set("valid_metric", valid_metric)

    task.set("train_dataset", train_dataset)
    task.set("valid_dataset", valid_dataset)

    return task


def symlink_best_model_path(trainer):
    wd = os.getcwd()
    best_model_path = None
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            best_model_path = callback.best_model_path
            break
    if best_model_path:
        save_checkpoint_path = osp.dirname(best_model_path)
        os.chdir(save_checkpoint_path)
        if osp.exists("best.ckpt"):
            _ = os.system(f"rm best.ckpt")
        _ = os.system(f"ln -s {best_model_path.split('/')[-1]} best.ckpt")
        os.chdir(wd)


def train(cfg, args):
    task = define_task(cfg, args)
    num_snapshots = cfg.train.get("num_snapshots") or 1
    if num_snapshots == 1:
        trainer = build_trainer(cfg, args, task)
        trainer.fit(task)
        symlink_best_model_path(trainer)
    else:
        for i in range(num_snapshots):
            if i > 0:
                # Need to rebuild optimizer and scheduler
                task.set(
                    "optimizer", builder.build_optimizer(cfg, task.model.parameters())
                )
                task.set("scheduler", builder.build_scheduler(cfg, task.optimizer))
            trainer = build_trainer(cfg, args, task, snapshot=i)
            trainer.fit(task)
            symlink_best_model_path(trainer)
