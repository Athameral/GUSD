import os
import utils
import time
import logging
import argparse
import importlib
import torch
import torch.distributed
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet.apis import set_random_seed, multi_gpu_test, single_gpu_test
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from models.utils import VERSION
from loaders.StreamingEval.gpu_tests import single_gpu_test_streaming


def evaluate(dataset, results, epoch, offset=0):
    metrics = dataset.evaluate(results, jsonfile_prefix='submission', offset=offset)
    return metrics

    # mAP = metrics['pts_bbox_NuScenes/mAP']
    # mATE = metrics['pts_bbox_NuScenes/mATE']
    # mASE = metrics['pts_bbox_NuScenes/mASE']
    # mAOE = metrics['pts_bbox_NuScenes/mAOE']
    # mAVE = metrics['pts_bbox_NuScenes/mAVE']
    # mAAE = metrics['pts_bbox_NuScenes/mAAE']
    # NDS = metrics['pts_bbox_NuScenes/NDS']

    # logging.info('--- Evaluation Results (Epoch %d) ---' % epoch)
    # logging.info('mAP: %.4f' % metrics['pts_bbox_NuScenes/mAP'])
    # logging.info('mATE: %.4f' % metrics['pts_bbox_NuScenes/mATE'])
    # logging.info('mASE: %.4f' % metrics['pts_bbox_NuScenes/mASE'])
    # logging.info('mAOE: %.4f' % metrics['pts_bbox_NuScenes/mAOE'])
    # logging.info('mAVE: %.4f' % metrics['pts_bbox_NuScenes/mAVE'])
    # logging.info('mAAE: %.4f' % metrics['pts_bbox_NuScenes/mAAE'])
    # logging.info('NDS: %.4f' % metrics['pts_bbox_NuScenes/NDS'])

    # return {
    #     'mAP': mAP,
    #     'mATE': mATE,
    #     'mASE': mASE,
    #     'mAOE': mAOE,
    #     'mAVE': mAVE,
    #     'mAAE': mAAE,
    #     'NDS': NDS,
    # }


def latency_test(model, data_loader, num_warmup=10, samples=500, log_interval=50):
    """Measure end-to-end latency by wrapping the whole forward pass with
    wall-clock time (same protocol as OPUS/scripts/timing.py).

    The measured window covers dataloader->GPU scatter, feature extraction,
    occupancy head and the occupancy/gaussian post-processing, i.e. the full
    ``model(...)`` call.
    """
    model.eval()
    times = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            if i >= num_warmup:
                times.append(elapsed)

            if (i + 1) % log_interval == 0:
                logging.info('[Latency] done %d/%d' % (i + 1, samples))
            if (i + 1) >= samples:
                break

    if not times:
        logging.info('[Latency] no samples measured (num_warmup >= samples).')
        return

    arr = torch.tensor(times) * 1000.0
    logging.info(
        '[Latency] FINAL (n=%d, warmup=%d)  '
        'avg=%.1fms  min=%.1fms  max=%.1fms  fps=%.1f'
        % (len(times), num_warmup, arr.mean().item(), arr.min().item(),
           arr.max().item(), 1000.0 / arr.mean().item()))


def main():
    parser = argparse.ArgumentParser(description='Validate a detector')
    parser.add_argument('--config', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--no_stream_eval', action='store_true')
    parser.add_argument('--offset', type=int, default=0, help='eval with GT at pred_idx + offset (for copy-paste evaluation)')
    parser.add_argument('--latency', action='store_true', help='latency measurement mode: no eval, no streaming, single GPU')
    parser.add_argument('--num_warmup', type=int, default=10, help='warmup iters for latency measurement')
    parser.add_argument('--samples', type=int, default=500, help='number of samples used for latency measurement')
    args = parser.parse_args()

    # latency mode: force single GPU, no DDP
    if args.latency:
        args.world_size = 1
        args.no_stream_eval = True

    # parse configs
    cfgs = Config.fromfile(args.config)

    # register custom module
    importlib.import_module('models')
    importlib.import_module('loaders')

    # MMCV, please shut up
    from mmcv.utils.logging import logger_initialized
    logger_initialized['root'] = logging.Logger(__name__, logging.WARNING)
    logger_initialized['mmcv'] = logging.Logger(__name__, logging.WARNING)

    # you need GPUs
    assert torch.cuda.is_available()

    # determine local_rank and world_size
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = str(args.world_size)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    if local_rank == 0:
        utils.init_logging(None, cfgs.debug)
    else:
        logging.root.disabled = True

    logging.info('Using GPU: %s' % torch.cuda.get_device_name(local_rank))
    torch.cuda.set_device(local_rank)

    if world_size > 1:
        logging.info('Initializing DDP with %d GPUs...' % world_size)
        dist.init_process_group('nccl', init_method='env://')

    logging.info('Setting random seed: 0')
    set_random_seed(0, deterministic=True)
    cudnn.benchmark = True

    logging.info('Loading validation set from %s' % cfgs.data.val.data_root)
    val_dataset = build_dataset(cfgs.data.val)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=args.batch_size,
        workers_per_gpu=cfgs.data.workers_per_gpu,
        num_gpus=world_size,
        dist=world_size > 1,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=6,
        persistent_workers=True,
        seed=0,
    )

    logging.info('Creating model: %s' % cfgs.model.type)
    model = build_model(cfgs.model)
    model.cuda()
    model.fp16_enabled = True

    if world_size > 1:
        model = MMDistributedDataParallel(model, [local_rank], broadcast_buffers=False)
    else:
        model = MMDataParallel(model, [0])

    logging.info('Loading checkpoint from %s' % args.weights)
    checkpoint = load_checkpoint(
        model, args.weights, map_location='cuda', strict=False,
        logger=logging.Logger(__name__, logging.ERROR)
    )

    if 'version' in checkpoint:
        VERSION.name = checkpoint['version']

    if world_size > 1:
        results = multi_gpu_test(model, val_loader, gpu_collect=False)
        eval_thread = None
    elif args.latency:
        latency_test(model, val_loader, num_warmup=args.num_warmup, samples=args.samples)
        logging.info('[Latency] measurement done, skipping eval.')
        return
    else:
        if not args.no_stream_eval:
            results, eval_thread = single_gpu_test_streaming(model, val_loader, data_size=len(val_dataset), local_rank=local_rank)
            logging.info('Started streaming evaluation thread. Running in background...')
        else:
            # breakpoint()
            results = single_gpu_test(model, val_loader)
            eval_thread = None

    if local_rank == 0:
        try:
            evaluate(val_dataset, results, -1, offset=args.offset)
        finally:
            if eval_thread is not None:
                eval_thread.join()


if __name__ == '__main__':
    main()
