import numpy as np
import torch
import torch.nn.functional as F


def torch_derangement(x):
    a = torch.arange(x)
    b = torch.randperm(x)
    while (a == b).sum().item() > 0:
        b = torch.randperm(x)
    return b


def apply_mixup(X, alpha):
    lam = np.random.beta(alpha, alpha, X.size(0))
    lam = np.max((lam, 1-lam), axis=0)
    index = torch_derangement(X.size(0))
    lam = torch.Tensor(lam).to(X.device)
    for dim in range(X.ndim - 1):
        lam = lam.unsqueeze(-1)
    X = lam * X + (1 - lam) * X[index]
    return X, index, lam


def rand_bbox(size, lam, margin=0):
    # lam is a vector
    B = size[0]
    assert B == lam.shape[0]
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_h = (H * cut_rat).astype(np.int)
    cut_w = (W * cut_rat).astype(np.int)
    # uniform
    if margin < 1 and margin > 0:
        h_margin = margin*H
        w_margin = margin*W
    else:
        h_margin = margin
        w_margin = margin
    cx = np.random.randint(0+h_margin, H-h_margin, B)
    cy = np.random.randint(0+w_margin, W-w_margin, B)
    #
    bbx1 = np.clip(cx - cut_h // 2, 0, H)
    bby1 = np.clip(cy - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_w // 2, 0, W)
    return bbx1, bby1, bbx2, bby2


def apply_cutmix(X, alpha, y=None):
    SEG = not isinstance(y, type(None))
    batch_size = X.size(0)
    lam = np.random.beta(alpha, alpha, batch_size)
    lam = np.max((lam, 1-lam), axis=0)
    x1, y1, x2, y2 = rand_bbox(X.size(), lam)
    index = torch_derangement(batch_size)
    for b in range(batch_size):
        X[b, ..., x1[b]:x2[b], y1[b]:y2[b]] = X[index[b], ..., x1[b]:x2[b], y1[b]:y2[b]]
        if SEG:
            y[b, ..., x1[b]:x2[b], y1[b]:y2[b]] = y[index[b], ..., x1[b]:x2[b], y1[b]:y2[b]]
    lam = 1. - ((x2 - x1) * (y2 - y1) / float((X.size(-1) * X.size(-2))))
    lam = torch.Tensor(lam).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    if SEG:
        return X, y, index, lam
    return X, index, lam


def rand_region(size, patch_size):
    H, W = size
    pH, pW = patch_size
    maxH = H - pH
    maxW = W - pW
    x1 = np.random.randint(0, maxH)
    y1 = np.random.randint(0, maxW)
    x2 = x1 + pH
    y2 = y1 + pW
    return x1, y1, x2, y2


def apply_resizemix(X, alphabeta, y=None):
    alpha, beta = alphabeta
    SEG = not isinstance(y, type(None))
    assert alpha > 0, 'alpha should be larger than 0'
    assert beta < 1, 'beta should be smaller than 1'
    batch_size = X.size(0)
    index = torch_derangement(batch_size)
    tau = np.random.uniform(alpha, beta, batch_size)
    lam = tau ** 2
    H, W = X.size()[2:]
    for b in range(batch_size):
        _tau = tau[b]
        patch_size = (int(H*_tau), int(W*_tau))
        resized_X = F.interpolate(X[index[b]].unsqueeze(0), size=patch_size, mode='bilinear', align_corners=False).squeeze(0)
        x1, y1, x2, y2 = rand_region((H, W), patch_size)
        X[b, ..., x1:x2, y1:y2] = resized_X
        if SEG:
            resized_y = F.interpolate(y[index[b]].unsqueeze(0), size=patch_size, mode='nearest').squeeze(0)
            y[b, ..., x1:x2, y1:y2] = resized_y
    lam = torch.Tensor(lam).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    if SEG:
        return X, y, index, lam
    return X, index, lam


MIX_FN = {'cutmix': apply_cutmix, 'mixup': apply_mixup, 'resizemix': apply_resizemix}
def apply_mixaug(X, y, mix):
    """
    mix is a dictionary
    Each key-value pair is a type of mix augmentation and the
    corresponding parameter(s) 
    If more than one key-value pair exists, then one will be randomly
    sampled.
    Currently, the parameters should be scalars.
    TODO: add support to sample parameters from a distribution.
    """
    mixer = np.random.choice([*mix])
    X, index, lam = MIX_FN[mixer](X, mix[mixer])
    return X, {
        'y1':  y,
        'y2':  y[index],
        'lam': lam.to(y.device)
    }


def apply_mixaug_seg(X, y, mix):
    ycls, yseg = y
    mixer = np.random.choice([*mix])
    if mixer in ['cutmix', 'resizemix']:
        X, yseg, index, lam = MIX_FN[mixer](X, mix[mixer], y=yseg)
    else:
        X, index, lam = MIX_FN[mixer](X, mix[mixer])
    return X, {
        'y1_cls':  ycls,
        'y2_cls':  ycls[index],
        'y1_seg':  yseg,
        # If using cutmix, segmentation ground truth has been edited
        # so just return the gt twice
        'y2_seg':  yseg if mixer in ['cutmix', 'resizemix'] else yseg[index],
        'lam': lam.to(X.device)
    }




