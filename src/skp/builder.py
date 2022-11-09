import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import data
from . import losses
from . import models 
from . import optim
from . import tasks


def get_name_and_params(base):
    name = getattr(base, 'name')
    params = getattr(base, 'params') or {}
    return name, params


def get_transform(base, transform, mode=None):
    if not base: return None
    transform = getattr(base, transform)
    if not transform: return None
    name, params = get_name_and_params(transform)
    if mode:
        params.update({'mode': mode})
    return getattr(data.transforms, name)(**params)


def build_transforms(cfg, mode):
    # 1-Resize
    resizer = get_transform(cfg.transform, 'resize')
    # 2-(Optional) Data augmentation
    augmenter = None
    if mode == "train":
        augmenter = get_transform(cfg.transform, 'augment')
    # 3-(Optional) Crop
    cropper = get_transform(cfg.transform, 'crop', mode=mode)
    # 4-Preprocess
    preprocessor = get_transform(cfg.transform, 'preprocess')
    return {
        'resize': resizer,
        'augment': augmenter,
        'crop': cropper,
        'preprocess': preprocessor
    }


def build_dataset(cfg, data_info, mode):
    dataset_class = getattr(data.datasets, cfg.data.dataset.name)
    dataset_params = cfg.data.dataset.params
    dataset_params.test_mode = mode != 'train'
    dataset_params = dict(dataset_params)
    if "FeatureDataset" not in cfg.data.dataset.name:
        transforms = build_transforms(cfg, mode)
        dataset_params.update(transforms)
    dataset_params.update(data_info)
    return dataset_class(**dataset_params)


def build_dataloader(cfg, dataset, mode):

    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    dataloader_params = {}
    dataloader_params['num_workers'] = cfg.data.num_workers
    dataloader_params['drop_last'] = mode == 'train'
    dataloader_params['shuffle'] = mode == 'train'
    dataloader_params["pin_memory"] = cfg.data.get("pin_memory", True)
    if mode in ('train', 'valid'):
        if mode == "train":
            dataloader_params['batch_size'] = cfg.train.batch_size
        elif mode == "valid":
            dataloader_params["batch_size"] = cfg.evaluate.get("batch_size") or cfg.train.batch_size
        sampler = None
        if cfg.data.get("sampler") and mode == 'train':
            name, params = get_name_and_params(cfg.data.sampler)
            sampler = getattr(data.samplers, name)(dataset, **params)
        if sampler:
            dataloader_params['shuffle'] = False
            if cfg.strategy == 'ddp':
                sampler = data.samplers.DistributedSamplerWrapper(sampler)
            dataloader_params['sampler'] = sampler
            print(f'Using sampler {sampler} for training ...')
        elif cfg.strategy == 'ddp':
            dataloader_params["shuffle"] = False
            dataloader_params['sampler'] = DistributedSampler(dataset, shuffle=mode=="train")
    else:
        assert cfg.strategy != "ddp", "DDP currently not supported for inference"
        dataloader_params['batch_size'] = cfg.evaluate.get("batch_size") or cfg.train.batch_size

    loader = DataLoader(dataset,
        **dataloader_params,
        worker_init_fn=worker_init_fn)
    return loader


def build_model(cfg):
    name, params = get_name_and_params(cfg.model)
    if cfg.model.params.get("cnn_params", None):
        cnn_params = cfg.model.params.cnn_params
        if cnn_params.get("load_pretrained_backbone", None):
            if "foldx" in cnn_params.load_pretrained_backbone:
                cfg.model.params.cnn_params.load_pretrained_backbone = cnn_params.load_pretrained_backbone.\
                    replace("foldx", f"fold{cfg.data.outer_fold}")
    print(f'Creating model <{name}> ...')
    model = getattr(models.engine, name)(**params)
    if 'backbone' in cfg.model.params:
        print(f'  Using backbone <{cfg.model.params.backbone}> ...')
    if 'pretrained' in cfg.model.params:
        print(f'  Pretrained : {cfg.model.params.pretrained}')
    if "load_pretrained" in cfg.model:
        import re
        if "foldx" in cfg.model.load_pretrained:
            cfg.model.load_pretrained = cfg.model.load_pretrained.replace("foldx", f"fold{cfg.data.outer_fold}")
        print(f"  Loading pretrained checkpoint from {cfg.model.load_pretrained}")
        weights = torch.load(cfg.model.load_pretrained, map_location=lambda storage, loc: storage)['state_dict']
        weights = {re.sub(r'^model.', '', k) : v for k,v in weights.items() if "loss_fn" not in k}
        model.load_state_dict(weights) 
    return model 


def build_loss(cfg):
    name, params = get_name_and_params(cfg.loss)
    print(f'Using loss function <{name}> ...')
    params = dict(params)
    if "pos_weight" in params:
        params["pos_weight"] = torch.tensor(params["pos_weight"])
    criterion = getattr(losses, name)(**params)
    return criterion


def build_scheduler(cfg, optimizer):
    # Some schedulers will require manipulation of config params
    # My specifications were to make it more intuitive for me
    name, params = get_name_and_params(cfg.scheduler)
    print(f'Using learning rate schedule <{name}> ...')

    if name == 'CosineAnnealingLR':
        # eta_min <-> final_lr
        # Set T_max as 100000 ... this is changed in on_train_start() method
        # of the LightningModule task 

        params = {
            'T_max': 100000,
            'eta_min': max(params.final_lr, 1.0e-8)
        }

    if name in ('OneCycleLR', 'CustomOneCycleLR'):
        # Use learning rate from optimizer parameters as initial learning rate
        lr_0 = cfg.optimizer.params.lr
        lr_1 = params.max_lr
        lr_2 = params.final_lr
        # lr_0 -> lr_1 -> lr_2 
        pct_start = params.pct_start
        params = {}
        params['steps_per_epoch'] = 100000 # see above- will fix in task
        params['epochs'] = cfg.train.num_epochs
        params['max_lr'] = lr_1
        params['pct_start'] = pct_start
        params['div_factor'] = lr_1 / lr_0 # max/init
        params['final_div_factor'] = lr_0 / max(lr_2, 1.0e-8) # init/final

    scheduler = getattr(optim, name)(optimizer=optimizer, **params)
    
    # Some schedulers might need more manipulation after instantiation
    if name in ('OneCycleLR', 'CustomOneCycleLR'):
        scheduler.pct_start = params['pct_start']

    # Set update frequency
    if name in ('OneCycleLR', 'CustomOneCycleLR', 'CosineAnnealingLR'):
        scheduler.update_frequency = 'on_batch'
    elif name in ('ReduceLROnPlateau'):
        scheduler.update_frequency = 'on_valid'
    else:
        scheduler.update_frequency = 'on_epoch'

    return scheduler


def build_optimizer(cfg, parameters):
    name, params = get_name_and_params(cfg.optimizer)
    print(f'Using optimizer <{name}> ...')
    optimizer = getattr(optim, name)(parameters, **params)
    return optimizer


def build_task(cfg, model):
    name, params = get_name_and_params(cfg.task)
    print(f'Building task <{name}> ...')
    return getattr(tasks, name)(cfg, model, **params)


