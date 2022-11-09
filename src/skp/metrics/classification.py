import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm

from functools import partial
from sklearn import metrics as skm 


def _roc_auc_score(t, p):
    try:
        return torch.tensor(skm.roc_auc_score(t, p))
    except Exception as e:
        print(e)
        return torch.tensor(0.5)


def _average_precision_score(t, p):
    try:
        return torch.tensor(skm.average_precision_score(t, p))
    except Exception as e:
        print(e)
        return torch.tensor(0)


class _BaseMetric(tm.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)

    def update(self, p, t):
        self.p.append(p)
        self.t.append(t)

    def compute(self):
        raise NotImplementedError


class _ScoreBased(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy()  # (N,) or (N,C)
        t = torch.cat(self.t, dim=0).cpu().numpy()  # (N,) or (N,C)
        if p.ndim == 1:
            # Binary classification
            return {f"{self.name}_mean": self.metric_func(t, p)}
        metrics_dict = {}
        for c in range(p.shape[1]):
            # Depends on whether it is multilabel or multiclass
            tmp_gt = t == c if t.ndim == 1 else t[:, c]
            metrics_dict[f"{self.name}{c}"] = self.metric_func(tmp_gt, p[:, c])
        metrics_dict[f"{self.name}_mean"] = np.mean([v for v in metrics_dict.values()])
        return metrics_dict


class _ClassBased(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy()  # (N,) or (N,C)
        t = torch.cat(self.t, dim=0).cpu().numpy()  # (N,) or (N,C)
        if p.ndim == 1:
            # Binary classification
            return {f"{self.name}": self.metric_func(t, p)}
        p = np.argmax(p, axis=1)
        return {f"{self.name}": self.metric_func(t, p)} 


class AUROC(_ScoreBased):

    name = "auc"
    def metric_func(self, t, p): return _roc_auc_score(t, p)


class AVP(_ScoreBased):

    name = "avp"
    def metric_func(self, t, p): return _average_precision_score(t, p)


class Accuracy(_ClassBased):

    name = "accuracy"
    def metric_func(self, t, p): return skm.accuracy_score(t, p)


class Kappa(_ClassBased):

    name = "kappa"
    def metric_func(self, t, p): return skm.cohen_kappa_score(t, p)


class QWK(_ClassBased):
    name = "qwk"
    def metric_func(self, t, p): return skm.cohen_kappa_score(t, p, weights="quadratic")


class CompetitionMetric(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu()  # (N,8)
        t = torch.cat(self.t, dim=0).cpu()  # (N,8)

        num_vals = p.shape[0] * p.shape[1]
        p = p.view(num_vals, -1).squeeze(1)
        t = t.view(num_vals, -1).squeeze(1)
        weights = np.asarray([1.] * num_vals)
        weights[7::8] = 7.
        weights[t.cpu() == 1] *= 2.0
        weights = torch.Tensor(weights).float().to(p.device)
        loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction="none")
        return {"comp_metric": (weights * loss).sum() / weights.sum()}