import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from monai.inferers import sliding_window_inference
from torchvision.transforms.functional import hflip, vflip

from . import builder
from . import optim
from . import metrics as pl_metrics
from .models.sequence import DualTransformerV2
from .data.mixaug import apply_mixaug, apply_mixaug_seg
from .data.mosaic import apply_mosaic
from .data.mosaic3d import apply_mosaic_3d


def crop_lt(x, crop_h, crop_w):
    """crop left top corner"""
    return x[..., 0:crop_h, 0:crop_w]


def crop_lb(x, crop_h, crop_w):
    """crop left bottom corner"""
    return x[..., -crop_h:, 0:crop_w]


def crop_rt(x, crop_h, crop_w):
    """crop right top corner"""
    return x[..., 0:crop_h, -crop_w:]


def crop_rb(x, crop_h, crop_w):
    """crop right bottom corner"""
    return x[..., -crop_h:, -crop_w:]


def center_crop(x, crop_h, crop_w):
    """make center crop"""

    center_h = x.shape[-2] // 2
    center_w = x.shape[-1] // 2
    half_crop_h = crop_h // 2
    half_crop_w = crop_w // 2

    y_min = center_h - half_crop_h
    y_max = center_h + half_crop_h + crop_h % 2
    x_min = center_w - half_crop_w
    x_max = center_w + half_crop_w + crop_w % 2

    return x[..., y_min:y_max, x_min:x_max]


class BaseTask(pl.LightningModule): 

    def __init__(self, cfg, model, mixaug=None):
        super().__init__()
        self.cfg = cfg
        self.model = model 
        self.mixaug = mixaug

        self.val_loss = []
        
        self.save_hyperparameters(cfg, ignore=['model'])
        
    def set(self, name, attr):
        if name == 'metrics':
            attr = nn.ModuleList(attr) 
        setattr(self, name, attr)
    
    def on_train_start(self): 
        for obj in ['optimizer','scheduler','loss_fn','metrics','valid_metric']:
            assert hasattr(self, obj)

        self.total_training_steps = self.trainer.num_training_batches * self.trainer.max_epochs 
        self.current_training_step = 0
        
        if isinstance(self.scheduler, optim.CosineAnnealingLR):
            self.scheduler.T_max = self.total_training_steps 

        if isinstance(self.scheduler, (optim.OneCycleLR, optim.CustomOneCycleLR)):
            self.scheduler.total_steps = self.total_training_steps
            self.scheduler.step_size_up = float(self.scheduler.pct_start * self.scheduler.total_training_steps) - 1
            self.scheduler.step_size_down = float(self.scheduler.total_training_steps - self.scheduler.step_size_up) - 1

    def _apply_mixaug(self, X, y):
        return apply_mixaug(X, y, self.mixaug)

    def training_step(self, batch, batch_idx):             
        X, y = batch
        if isinstance(self.mixaug, dict):
            X, y = self._apply_mixaug(X, y)
        p = self.model(X) 
        loss = self.loss_fn(p, y)
        self.log('loss', loss) 
        self.current_training_step += 1
        return loss

    def validation_step(self, batch, batch_idx): 
        X, y = batch
        p = self.model(X)
        loss = self.loss_fn(p, y)
        self.val_loss += [loss]
        if isinstance(self.model, DualTransformerV2):
            p = p[1]
            y = y[1]
        for m in self.metrics:
            m.update(p, y)
        return loss

    def predict_act_fn(self, x):
        act_fn = self.cfg.get("inference_act_fn", None)
        if act_fn == "sigmoid":
            return torch.sigmoid(x)
        elif act_fn == "softmax":
            return torch.softmax(x, dim=1)
        else:
            return x

    def predict_step(self, batch, batch_idx): 
        # dataloader should return name
        X, y, name = batch
        tta_ops = self.cfg.get("tta", None)
        if tta_ops == "5crop":
            crop_h, crop_w = self.cfg.tta_5crop_size
            p = torch.stack([self.predict_act_fn(self.model(crop_func(X, crop_h, crop_w)))
                             for crop_func in
                             [center_crop, crop_lt, crop_rt, crop_lb, crop_rb]]).mean(0)
        elif tta_ops == "flip":
            p = torch.stack([self.predict_act_fn(self.model(X)),
                             self.predict_act_fn(self.model(hflip(X))),
                             self.predict_act_fn(self.model(vflip(X)))]).mean(0)
        else:
            p = self.predict_act_fn(self.model(X))
        return p, name

    def validation_epoch_end(self, *args, **kwargs):
        metrics = {}
        for m in self.metrics:
            metrics.update(m.compute())
        metrics['val_loss'] = torch.stack(self.val_loss).mean() ; self.val_loss = []
        max_strlen = max([len(k) for k in metrics.keys()])

        if isinstance(self.valid_metric, list):
            metrics['vm'] = torch.sum(torch.stack([metrics[_vm.lower()].cpu() for _vm in self.valid_metric]))
        else:
            metrics['vm'] = metrics[self.valid_metric.lower()]

        self.log_dict(metrics)

        for m in self.metrics: m.reset()

        if self.global_rank == 0:
            print('\n========')
            for k,v in metrics.items(): 
                print(f'{k.ljust(max_strlen)} | {v.item():.4f}')

    def configure_optimizers(self):
        lr_scheduler = {
            'scheduler': self.scheduler,
            'interval': 'step' if self.scheduler.update_frequency == 'on_batch' else 'epoch'
        }
        if isinstance(self.scheduler, optim.ReduceLROnPlateau): 
            lr_scheduler['monitor'] = self.valid_metric
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': lr_scheduler
            }

    def train_dataloader(self):
        return builder.build_dataloader(self.cfg, self.train_dataset, 'train')

    def val_dataloader(self):
        return builder.build_dataloader(self.cfg, self.valid_dataset, 'valid')


class ClassificationTask(BaseTask): pass


class SegmentationTask(ClassificationTask): 

    def __init__(self, *args, **kwargs):
        self.mosaic = kwargs.pop("mosaic", False)
        self.multislice_pred = kwargs.pop("multislice_pred", False)
        super().__init__(*args, **kwargs)
        if self.mosaic:
            assert not isinstance(self.mixaug, dict), "Mosaic and mix augmentation cannot be used at the same time"

    def training_step(self, batch, batch_idx):             
        X, y = batch
        if self.mosaic:
            X, y = apply_mosaic(X, y)
        if isinstance(self.mixaug, dict):
            X, y = self._apply_mixaug(X, y)
        p = self.model(X) 
        loss = self.loss_fn(p, y)
        self.log('loss', loss) 
        self.current_training_step += 1
        return loss

    def _apply_mixaug(self, X, y):
        return apply_mixaug_seg(X, y, self.mixaug)

    def validation_step(self, batch, batch_idx):
        X, y = batch
        p = self.model(X) 
        if self.multislice_pred: 
            p = p[:, 3:6]
        # p.shape = (N,C,H,W)
        loss = self.loss_fn(p, y)
        self.val_loss += [loss]
        # Downsample to increase speed/decreases mem usage
        #p = F.interpolate(p, scale_factor=0.5)
        #y = F.interpolate(y, scale_factor=0.5)
        for m in self.metrics: m.update(p, y)


class SegmentationTask3D(SegmentationTask):

    def __init__(self, *args, **kwargs):
        self.roi_size = kwargs.pop("roi_size", [128, 128, 128])
        self.chunk_validation = kwargs.pop("chunk_validation", True)
        self.chunk_inference = kwargs.pop("chunk_inference", False)
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):             
        X, y = batch
        if self.mosaic:
            X, y = apply_mosaic_3d(X, y)
        if isinstance(self.mixaug, dict):
            X, y = self._apply_mixaug(X, y)
        p = self.model(X) 
        loss = self.loss_fn(p, y)
        self.log('loss', loss) 
        self.current_training_step += 1
        return loss
    # def pad(self, x):
    #     N, C, Z, H, W = x.size()
    #     scale_factor = Z // 32 + 1
    #     padding_needed = scale_factor * 32 - Z
    #     padding = torch.zeros((N, C, padding_needed, H, W)).float().to(x.device)
    #     padding[...] = x.min()
    #     x = torch.cat([x, padding], dim=2) 
    #     return x

    def validation_step(self, batch, batch_idx):
        # Entire volume is returned 
        X, y = batch

        if self.chunk_validation:
            p = self.model(X)
        else:
            p = sliding_window_inference(inputs=X, roi_size=self.roi_size, sw_batch_size=X.size(0), predictor=self.model)

        loss = self.loss_fn(p, y)
        self.val_loss += [loss]
        # p.shape = (N, C, Z, H, W)
        for m in self.metrics: 
            for i in range(p.size(0)):
                m.update(p[i].unsqueeze(0), y[i].unsqueeze(0))

    def predict_step(self, batch, batch_idx):
        # Easier to just write a separate script outside of PyTorch-Lightning
        pass


