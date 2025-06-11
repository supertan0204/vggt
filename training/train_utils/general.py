import torch
import torch.nn as nn
import os
import math
from typing import Union, Optional
import logging
from iopath.common.file_io import g_pathmgr



class AverageMeter:
    """Computes and stores the average and current value.
    Args:
        name (str): Name of the metric being tracked
        device (torch.device, optional): Device for tensor operations. Defaults to None.
        fmt (str): Format string for displaying values. Defaults to ":f"
    """

    def __init__(self, name: str, device: Optional[torch.device] = None, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, val, n=1):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        """String representation showing current and average values."""
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    @property
    def value(self) -> float:
        """Get the current value."""
        return self.val

    @property
    def average(self) -> float:
        """Get the running average."""
        return self.avg



def safe_makedirs(path: str):
    if not path:
        logging.warning("safe_makedirs called with an empty path. No operation performed.")
        return False

    try:
        os.makedirs(path, exist_ok=True)
        logging.info(f"Directory '{path}' now exists.")
        return True
    except OSError as e:
        logging.error(f"Failed to create directory '{path}'. Reason: {e}")
        raise
    except Exception as e:
        # Catch any other unexpected errors.
        logging.error(f"An unexpected error occurred while creating directory '{path}'. Reason: {e}")
        raise
