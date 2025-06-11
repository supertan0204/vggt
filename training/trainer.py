# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
# Plan: This file should provide a trainer class, which provides
# 1. The init of DDP
# 2. The init of optimizers, tb, timers, and so on
# 3. A basic training framework (especially for finetuning)
#       self._train_epoch_
#       self._process_batch_
#       self._step_
# ETA: hope I can finish this before June 8th
'''
import contextlib

import copy
import functools
import gc
import json
import logging
import math
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

import random
import string
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision

from einops import rearrange
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from PIL import Image

from datetime import timedelta

#
from train_utils.general import AverageMeter, safe_makedirs
from train_utils.logging import setup_logging
from train_utils.distributed import get_machine_local_and_dist_rank

class Trainer:
    """
    Trainer supporting the DDP training strategies.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        resume_checkpoint_path: Optional[str] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
    ):
        self.resume_checkpoint_path = resume_checkpoint_path

        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        # self.checkpoint_conf = TrainerCheckpointConf(**checkpoint).infer_missing()

        # hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.optim_conf = optim

        self.where = 0.0
        self.seed_value = seed_value

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        safe_makedirs(self.logging_conf.log_dir)

        import pdb;pdb.set_trace()
