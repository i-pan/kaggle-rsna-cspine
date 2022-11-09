# Optimizers
from torch.optim import (
    Adam,
    AdamW,
    SGD,
    RMSprop
)

try:
    from bitsandbytes.optim import AdamW8bit
except ImportError:
    pass

from .radam import RAdam
from .madgrad import MADGRAD

# Schedulers
from torch.optim.lr_scheduler import (
    OneCycleLR,
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

from .onecycle import CustomOneCycleLR
