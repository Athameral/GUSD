# Copyright (c) OpenMMLab. All rights reserved.
# modified from megvii-bevdepth.
import math
import os
from copy import deepcopy

import torch
from mmcv.runner import load_state_dict
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook
from .utils import is_parallel
from mmdet3d.models import build_model
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

__all__ = ['ModelEMA2']


class ModelEMA2:
    """Model Exponential Moving Average from https://github.com/rwightman/
    pytorch-image-models Keep a moving average of everything in the model
    state_dict (parameters and buffers).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/
    ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training
    schemes to perform well.
    This class is sensitive where it is initialized in the sequence
    of model init, GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0,ema_model_cfg=None):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        assert ema_model_cfg is not None, 'ema_model_cfg is None'
        self.ema_model = self.clone_model(model,ema_model_cfg).eval()

        self.ema = self.ema_model.module.module if is_parallel(
            self.ema_model.module) else self.ema_model.module
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)
        k=1

    def clone_model(self, model,ema_model_cfg):
        """
        Clone a model by copying its state_dict.
        """
        cloned_model = build_model(ema_model_cfg)
        if is_parallel(model.module):
            if type(model)==MMDistributedDataParallel():
                cloned_model=type(model)(cloned_model,[0],find_unused_parameters=True)
            else:
                cloned_model=type(model)(cloned_model,[0])
        
        cloned_model.load_state_dict(model.state_dict())

        return cloned_model

    def update(self, trainer, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(
                model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()


@HOOKS.register_module()
class MEGVIIEMAHook2(Hook):
    """EMAHook used in BEVDepth.

    Modified from https://github.com/Megvii-Base
    Detection/BEVDepth/blob/main/callbacks/ema.py.
    """

    def __init__(self, init_updates=0, decay=0.9990, resume=None,ema_model_cfg=None):
        super().__init__()
        self.init_updates = init_updates
        self.resume = resume
        self.decay = decay
        self.ema_model_cfg = ema_model_cfg

    def before_run(self, runner):
        from torch.nn.modules.batchnorm import SyncBatchNorm

        bn_model_list = list()
        bn_model_dist_group_list = list()
        for model_ref in runner.model.modules():
            if isinstance(model_ref, SyncBatchNorm):
                bn_model_list.append(model_ref)
                bn_model_dist_group_list.append(model_ref.process_group)
                model_ref.process_group = None
        runner.ema_model = ModelEMA2(runner.model, self.decay,ema_model_cfg=self.ema_model_cfg)

        for bn_model, dist_group in zip(bn_model_list,
                                        bn_model_dist_group_list):
            bn_model.process_group = dist_group
        runner.ema_model.updates = self.init_updates

        if self.resume is not None:
            runner.logger.info(f'resume ema checkpoint from {self.resume}')
            cpt = torch.load(self.resume, map_location='cpu')
            load_state_dict(runner.ema_model.ema, cpt['state_dict'])
            runner.ema_model.updates = cpt['updates']

    def after_train_iter(self, runner):
        runner.ema_model.update(runner, runner.model.module)

    def after_train_epoch(self, runner):
        self.save_checkpoint(runner)

    @master_only
    def save_checkpoint(self, runner):
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            'epoch': runner.epoch,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = f'epoch_{runner.epoch+1}_ema.pth'
        save_path = os.path.join(runner.work_dir, save_path)
        torch.save(ema_checkpoint, save_path)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')
