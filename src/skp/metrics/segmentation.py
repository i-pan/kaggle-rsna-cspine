import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics as tm


def _dice_score(p, t, threshold=0.5):
    """
    Input shape = (N, C, H*W[*Z])
    Output shape = (N, C), where element i,j is the Dice score for
      class Cj of sample Ni
    """
    p = (p >= threshold)
    scores = (2*(p*t).sum(-1)/((p+t).sum(-1)))
    return scores


class _BaseMetric(tm.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("metric", default=[], dist_reduce_fx=None)

    def update(self, p, t):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError


class DSC(_BaseMetric):

    def __init__(self, *args, **kwargs):
        self.thresholds = torch.linspace(0.1, 0.9, 9)
        self.activation = kwargs.pop("activation", "sigmoid")
        self.include_nan_to_1 = kwargs.pop("include_nan_to_1", False)
        super().__init__(*args, **kwargs)

    def update(self, p, t):
        # p.shape = (N, C, Z, H, W)
        # t.shape = (N, Z, H, W)
        N, C, = p.shape[:2]
        self.C = C
        if self.activation == "sigmoid":
            p = torch.sigmoid(p)
        elif self.activation == "softmax":
            p = torch.softmax(p, dim=1)
        p = p.reshape(N, C, -1)
        t = t.reshape(N, C, -1)
        dsc = torch.stack([_dice_score(p, t, thresh) for thresh in self.thresholds], dim=2)
        # dsc.shape = (N, C, num_thresholds)
        self.metric.append(dsc)

    def compute(self):
        dsc = torch.cat(self.metric, dim=0)
        dsc_empty1 = torch.nan_to_num(dsc, 1.0)

        metrics_dict = {}

        num_classes = self.C

        ##############
        # IGNORE NAN #
        ##############
        dsc_total = 0.
        for i in range(num_classes):
            best_dsc = -1.
            best_thresh = 0.
            for j in range(len(self.thresholds)):
                dsc_subset = dsc[:,i,j]
                # Only include DSC values that are not NaN
                dsc_at_thresh = dsc_subset[~dsc_subset.isnan()].mean()
                if dsc_at_thresh > best_dsc:
                    best_dsc = dsc_at_thresh
                    best_thresh = self.thresholds[j]
            metrics_dict[f"dsc_ignore_{i:02d}"] = best_dsc
            dsc_total += best_dsc
            metrics_dict[f"thr_ignore_{i:02d}"] = best_thresh
            metrics_dict["dsc_ignore_mean"] = dsc_total / num_classes
       
        ############
        # NAN TO 1 #
        ############
        if self.include_nan_to_1:
            dsc_total = 0.
            for i in range(num_classes):
                max_dsc = dsc_empty1[:,i].mean(0).max()
                best_thr = self.thresholds[dsc_empty1[:,i].mean(0).argmax().item()]
                metrics_dict[f"dsc_empty1_{i:02d}"] = max_dsc
                dsc_total += max_dsc
                metrics_dict[f"thr_empty1_{i:02d}"] = best_thr
                metrics_dict["dsc_empty1_mean"] = dsc_total / num_classes

        metrics_dict = {k : torch.tensor(v) if not isinstance(v, torch.Tensor) else v for k,v in metrics_dict.items()}
        
        return metrics_dict