import torch
import torch.distributed as dist
from tqdm import tqdm

import logging
import queue
import threading
import time


def single_gpu_test_streaming(model, data_loader, data_size=None, local_rank=0, tqdm_position=1):
    """Run single-GPU inference in a background thread, returning a queue for streaming consumption.

    Args:
        model: The model to evaluate.
        data_loader: DataLoader yielding input batches.
        data_size: Number of batches (used for tqdm total). If None, inferred from len(data_loader).
        local_rank: CUDA device index.
        tqdm_position: tqdm position offset.

    Returns:
        tuple: (queue.Queue, threading.Thread) — the queue receives per-sample result dicts;
               the caller MUST join() the thread after draining the queue.
    """
    if data_size is None:
        try:
            data_size = len(data_loader)
        except TypeError:
            data_size = None  # tqdm will run without a total

    logging.info('Starting single_gpu_test_streaming...')
    result_queue = queue.Queue(maxsize=100)
    logging.info(f'Setting queue maxsize to {result_queue.maxsize} for streaming evaluation...')

    def _worker():
        sample_idx = 0
        try:
            torch.cuda.set_device(local_rank)
            model.eval()
            for data in tqdm(data_loader, total=data_size, position=tqdm_position,
                             disable=(tqdm_position is None),
                             leave=True, desc='Inference', ncols=80):
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **data)

                if isinstance(result, list):
                    batch_results = result
                else:
                    batch_results = [result]

                for result_item in batch_results:
                    result_queue.put(result_item)
                    sample_idx += 1
                    # logging.info('Processed %d/%d samples' % (sample_idx, len(data_loader.dataset)))
        except Exception as e:
            result_queue.put(e)
        finally:
            result_queue.put(None)

    worker = threading.Thread(target=_worker, name='single-gpu-test-streaming', daemon=True)
    worker.start()
    return result_queue, worker


def multi_gpu_test_streaming(model, data_loader, data_size=None, local_rank=0, tqdm_position=1):
    """Run distributed multi-GPU inference in a background thread, returning a queue.

    Modeled after mmcv's official ``multi_gpu_test``: each rank iterates its
    DistributedSampler partition, then ``dist.gather_object`` collects results
    to rank 0 with ``zip(*gathered)`` ordering.

    Difference from official: runs in a background thread and returns
    ``(queue.Queue, threading.Thread)`` so the consumer can overlap
    gather-communication with evaluation, just like ``single_gpu_test_streaming``
    does for inference.

    Args:
        model: The model to evaluate.
        data_loader: DataLoader (with DistributedSampler).
        data_size: Number of batches per rank (tqdm total). None → auto.
        local_rank: CUDA device index.
        tqdm_position: tqdm position offset.

    Returns:
        tuple: (queue.Queue, threading.Thread) — rank 0's queue receives
               per-sample result dicts followed by a ``None`` sentinel.
               Caller **MUST** ``join()`` the thread after draining.
    """
    if data_size is None:
        try:
            data_size = len(data_loader)
        except TypeError:
            data_size = None

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    dataset = data_loader.dataset

    # Only rank 0 shows progress bar to avoid interleaved output
    if world_size > 1 and rank != 0:
        tqdm_position = None  # suppress tqdm on non-rank-0

    time.sleep(2)  # Prevent deadlock, consistent with official mmcv multi_gpu_test

    result_queue = queue.Queue(maxsize=0)

    def _worker():
        try:
            torch.cuda.set_device(local_rank)
            model.eval()
            results = []

            for data in tqdm(data_loader, total=data_size,
                             position=tqdm_position,
                             disable=(tqdm_position is None),
                             leave=True, desc='Inference', ncols=80):
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **data)
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)

            # --- gather results from all ranks (collective: every rank calls) ---
            if dist.is_initialized() and world_size > 1:
                if rank == 0:
                    gathered = [None] * world_size
                else:
                    gathered = None
                dist.gather_object(results, gathered, dst=0)

                if rank == 0:
                    # DistributedSampler: rank 0 → [0,w,2w…], rank 1 → [1,1+w…]
                    # zip(*gathered) → [(r0₀,r1₀), (r0₁,r1₁), …] → flatten → original order
                    ordered = []
                    for tup in zip(*gathered):
                        ordered.extend(list(tup))
                    # Trim padding samples from the last incomplete batch
                    results = ordered[:len(dataset)]
                else:
                    results = []

            # --- feed results into queue (rank 0 only) ---
            if rank == 0:
                for item in results:
                    result_queue.put(item)
        except Exception as e:
            if rank == 0:
                result_queue.put(e)
        finally:
            if rank == 0:
                result_queue.put(None)

    worker = threading.Thread(target=_worker, name='multi-gpu-test-streaming', daemon=True)
    worker.start()
    return result_queue, worker