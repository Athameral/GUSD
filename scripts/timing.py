import os
import sys
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path)

import time
import logging
import argparse
import importlib
import torch
import torch.distributed
import torch.backends.cudnn as cudnn
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model

from utils import init_logging


def main():
    parser = argparse.ArgumentParser(description='Measure end-to-end latency')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--num_warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--log-interval', type=int, default=50, help='interval of logging')
    parser.add_argument('--override', nargs='+', action=DictAction)
    args = parser.parse_args()

    # parse configs
    cfgs = Config.fromfile(args.config)
    if args.override is not None:
        cfgs.merge_from_dict(args.override)

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)
    init_logging(None, cfgs.debug)

    # you need GPUs
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    logging.info('Using GPU: %s' % torch.cuda.get_device_name(0))
    torch.cuda.set_device(0)

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=0,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()

    assert torch.cuda.device_count() == 1
    model = MMDataParallel(model, [0])

    logging.info('Loading checkpoint from %s' % args.weights)
    load_checkpoint(
        model, args.weights, map_location='cuda', strict=False,
        logger=logging.Logger(__name__, logging.ERROR)
    )
    model.eval()

    # Wall-clock timing wrapping the whole forward pass, so the measured
    # window covers dataloader->GPU scatter, feature extraction, the
    # occupancy head and the occupancy/gaussian post-processing. This is
    # the same protocol as OPUS/scripts/timing.py.
    pure_inf_time = 0
    times = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= args.num_warmup:
                pure_inf_time += elapsed
                times.append(elapsed)
                if len(times) % args.log_interval == 0:
                    fps = len(times) / pure_inf_time
                    print(f'Done sample [{i + 1:<3}/ {args.samples}], '
                        f'fps: {fps:.1f} sample / s')

            if (i + 1) == args.samples:
                break

    if times:
        arr = torch.tensor(times) * 1000.0
        print(
            f'[Latency] FINAL (n={len(times)}, warmup={args.num_warmup})  '
            f'avg={arr.mean().item():.1f}ms  min={arr.min().item():.1f}ms  '
            f'max={arr.max().item():.1f}ms  '
            f'fps={len(times) / pure_inf_time:.1f} sample / s'
        )


if __name__ == '__main__':
    main()
