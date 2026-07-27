import bisect
import os.path as osp

import mmcv
import torch.distributed as dist
from mmcv.runner import EvalHook as BaseEvalHook
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS
from torch.nn.modules.batchnorm import _BatchNorm

from .gpu_tests import (
    single_gpu_test_streaming,
    multi_gpu_test_streaming,
)

def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals

@HOOKS.register_module()
class StreamingEvalHook(BaseEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(StreamingEvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return


        # Changed results to self.results so that MMDetWandbHook can access
        # the evaluation results and log them to wandb.
        result_queue, worker = single_gpu_test_streaming(runner.model, self.dataloader)
        self.latest_results = result_queue
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        try:
            key_score = self.evaluate(runner, result_queue)
        finally:
            worker.join()
        # the key_score may be `None` so it needs to skip the action to save
        # the best checkpoint
        if self.save_best and key_score:
            self._save_ckpt(runner, key_score)


@HOOKS.register_module()
class DistStreamingEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None, **kwargs):
        super(DistStreamingEvalHook, self).__init__(*args, **kwargs)
        self.latest_results = None

        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Keep consistent with DistEvalHook: sync BN buffers from rank 0.
        if self.broadcast_bn_buffer:
            model = runner.model
            for _, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        print("before multi gpu streaming test")
        result_queue, worker = multi_gpu_test_streaming(runner.model, self.dataloader)
        print("after multi gpu streaming test")
        self.latest_results = result_queue

        rank, _ = get_dist_info()
        if rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            try:
                key_score = self.evaluate(runner, result_queue)
            finally:
                worker.join()
            if self.save_best and key_score:
                self._save_ckpt(runner, key_score)
        else:
            worker.join()