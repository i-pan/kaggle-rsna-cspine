import numpy as np
import pickle
import random
import torch

from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


class FeatureDataset(Dataset):

    def __init__(self,
                 inputs,
                 labels,
                 seq_len,
                 exam_level_label=False,
                 resample=False,
                 test_mode=False,
                 return_name=False,
                 reverse=False,
                 normalize=False,
                 noise=None):
        self.inputs = inputs
        self.labels = labels
        self.seq_len = seq_len
        self.exam_level_label = exam_level_label
        self.resample = resample
        self.reverse = reverse
        self.normalize = normalize
        self.test_mode = test_mode
        self.noise = noise
        self.return_name = return_name

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        X, y = np.load(self.inputs[i]), self.labels[i]

        if not self.test_mode and self.reverse and random.random() > 0.5:
            X = np.ascontiguousarray(X[::-1])
            if not self.exam_level_label:
                y = np.ascontiguousarray(y[::-1])

        if self.noise and not self.test_mode:
            X *= np.random.normal(loc=1, scale=self.noise, size=X.shape)

        truncate_or_resample = not self.test_mode or self.resample

        if len(X) > self.seq_len and truncate_or_resample:
            if self.resample:
                # Resample using nearest interpolation
                scale = self.seq_len/len(X)
                X = zoom(X, [scale, 1.], order=0, prefilter=False)
                if not self.exam_level_label:
                    y = zoom(y, [scale], order=0, prefilter=False)
            else:
                # Truncate
                start = np.random.randint(0, len(X)-self.seq_len)
                X = X[start:start+self.seq_len]
                if not self.exam_level_label:
                    y = y[start:start+self.seq_len]

        if len(X) < self.seq_len and truncate_or_resample:
            diff = self.seq_len-len(X)
            mask = np.asarray([1]*len(X) + [0]*diff)
            padding = np.zeros_like(X[0])
            padding = np.expand_dims(padding, axis=0)
            padding = np.concatenate([padding]*diff, axis=0)
            X = np.concatenate([X, padding], axis=0)
            if not self.exam_level_label:
                y = np.concatenate([y, [0]*diff])
        else:
            mask = np.asarray([1]*len(X))

        X = torch.tensor(X).float()
        if self.normalize:
            X = torch.nn.functional.normalize(X, dim=1)
        y = torch.tensor(y)
        mask = torch.tensor(mask).long()

        if self.return_name:
            return (X, mask), self.inputs[i]

        if self.exam_level_label:
            return (X, mask), y

        return (X, mask), (y, mask)


class DualFeatureDataset(Dataset):

    def __init__(self,
                 inputs,
                 labels,
                 seq_len,
                 resample=False,
                 test_mode=False,
                 return_name=False,
                 reverse=False,
                 normalize=False,
                 noise=None):
        self.inputs = inputs
        self.labels = labels
        self.seq_len = seq_len
        self.resample = resample
        assert resample, "This dataset currently only supports resampling to a fixed sequence length"
        self.reverse = reverse
        self.normalize = normalize
        self.test_mode = test_mode
        self.noise = noise
        self.return_name = return_name

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        X = np.load(self.inputs[i])

        with open(self.labels[i], "rb") as f:
            y = pickle.load(f)

        if not self.test_mode and self.reverse and random.random() > 0.5:
            X = np.ascontiguousarray(X[::-1])
            y["sequence"] = np.ascontiguousarray(y["sequence"][::-1])

        if self.noise and not self.test_mode:
            X *= np.random.normal(loc=1, scale=self.noise, size=X.shape)

        truncate_or_resample = not self.test_mode or self.resample

        if len(X) > self.seq_len and truncate_or_resample:
            if self.resample:
                # Resample using nearest interpolation
                scale = self.seq_len/len(X)
                X = zoom(X, [scale, 1.], order=0, prefilter=False)
                y["sequence"] = zoom(y["sequence"], [scale], order=0, prefilter=False)
            else:
                # Truncate
                start = np.random.randint(0, len(X)-self.seq_len)
                X = X[start:start+self.seq_len]
                y["sequence"] = y["sequence"][start:start+self.seq_len]

        if len(X) < self.seq_len and truncate_or_resample:
            diff = self.seq_len-len(X)
            mask = np.asarray([1]*len(X) + [0]*diff)
            padding = np.zeros_like(X[0])
            padding = np.expand_dims(padding, axis=0)
            padding = np.concatenate([padding]*diff, axis=0)
            X = np.concatenate([X, padding], axis=0)
            y["sequence"] = np.concatenate([y["sequence"], [0]*diff])
        else:
            mask = np.asarray([1]*len(X))

        X = torch.tensor(X).float()
        if self.normalize:
            X = torch.nn.functional.normalize(X, dim=1)
        for k, v in y.items():
            y[k] = torch.tensor(v)

        mask = torch.tensor(mask).long()

        if self.return_name:
            return (X, mask), self.inputs[i]

        return (X, mask), (y["sequence"], y["exam"])