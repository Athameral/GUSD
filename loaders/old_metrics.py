import os
from abc import ABC, abstractmethod
import numpy as np
from sklearn.neighbors import KDTree
from termcolor import colored
from functools import reduce
from typing import Iterable, Optional, List, Dict, Any, Union
import logging

import torch

np.seterr(divide='ignore', invalid='ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ---------------------------------------------------------------------------
# Pre-defined class-name tables
# ---------------------------------------------------------------------------
OCC3D_CLASS_NAMES_18 = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free',
]

OCC3D_CLASS_NAMES_2 = ['non-free', 'free']

OPENOCC_CLASS_NAMES = [
    'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free',
]


def pcolor(string, color, on_color=None, attrs=None):
    """
    Produces a colored string for printing

    Parameters
    ----------
    string : str
        String that will be colored
    color : str
        Color to use
    on_color : str
        Background color to use
    attrs : list of str
        Different attributes for the string

    Returns
    -------
    string: str
        Colored string
    """
    return colored(string, color, on_color, attrs)


def getCellCoordinates(points, voxelSize):
    return (points / voxelSize).astype(np.int)


def getNumUniqueCells(cells):
    M = cells.max() + 1
    return np.unique(cells[:, 0] + M * cells[:, 1] + M ** 2 * cells[:, 2]).shape[0]


# ===================================================================
#  Abstract Base Class
# ===================================================================

class BaseMetric(ABC):
    """Abstract base class for occupancy / segmentation metrics.

    **All internal computation uses PyTorch tensors** so metrics can run
    on CPU or GPU transparently.

    Core interface:
      - ``update(pred, gt, **kwargs)``  – accumulate one batch
      - ``compute()``                    – return the final metric dict
      - ``to(device)``                   – move internal state to cpu/gpu
      - ``reset()``                      – clear accumulated state
    """

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        device: str = 'cpu',
    ):
        self.num_classes = num_classes
        self.class_names = class_names
        self.device = device
        self.cnt = 0
        self._hist = torch.zeros(
            num_classes, num_classes, dtype=torch.float64, device=device,
        )

    # ---- device management ----
    def to(self, device: str) -> 'BaseMetric':
        """Move all internal tensors to ``device`` (``'cpu'`` or ``'cuda'``).

        Subclasses that hold additional tensors should override and chain
        ``super().to(device)``.
        """
        self._hist = self._hist.to(device)
        self.device = device
        return self

    # ---- accumulate / finalise (abstract) ----
    @abstractmethod
    def update(self, pred, gt, **kwargs) -> None:
        """Accumulate one sample / batch into the running confusion matrix.

        Parameters
        ----------
        pred : np.ndarray or torch.Tensor
            Predicted occupancy labels.
        gt : np.ndarray or torch.Tensor
            Ground-truth occupancy labels.
        **kwargs
            Optional masks / extra info.
        """
        ...

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """Return the final metric(s) computed from all accumulated data.

        Returns
        -------
        dict
            Dictionary with at least ``'mIoU'``.
        """
        ...

    # ---- reset ----
    def reset(self) -> None:
        """Clear the accumulated confusion matrix and sample counter."""
        self._hist = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.float64,
            device=self.device,
        )
        self.cnt = 0

    # ---- shared helpers (torch-based) ----
    @staticmethod
    def _as_tensor(x, device: str = 'cpu') -> torch.Tensor:
        """Convert ``x`` (ndarray or Tensor) to a torch Tensor on *device*.

        Uses ``torch.as_tensor`` so the original memory is shared when
        possible (no copy for same-device tensors).
        """
        return torch.as_tensor(x, device=device)

    @staticmethod
    def _to_1d_long(t: torch.Tensor) -> torch.Tensor:
        """Flatten and cast to ``int64`` for bincount indexing."""
        return t.flatten().long()

    @staticmethod
    def hist_info(n_cl: int, pred: torch.Tensor, gt: torch.Tensor):
        """Build a confusion matrix using ``torch.bincount``.

        Parameters
        ----------
        n_cl : int
            Number of classes.
        pred : torch.Tensor  (1-D, int)
            Predicted labels.
        gt : torch.Tensor    (1-D, int)
            Ground-truth labels.

        Returns
        -------
        hist : torch.Tensor  shape (n_cl, n_cl)
        correct : int
        labeled : int
        """
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)          # exclude ignore labels
        labeled = k.sum().item()
        correct = (pred[k] == gt[k]).sum().item()
        flat_idx = n_cl * gt[k].long() + pred[k].long()
        hist = torch.bincount(
            flat_idx, minlength=n_cl ** 2,
        ).reshape(n_cl, n_cl).to(dtype=torch.float64)
        return hist, correct, labeled

    @staticmethod
    def per_class_iu(hist: torch.Tensor) -> torch.Tensor:
        """Per-class IoU from a confusion matrix (torch Tensor).

        Returns a tensor of shape ``(n_cl,)``; classes with zero GT get
        ``nan``.
        """
        diag = hist.diag()
        sum_row = hist.sum(dim=1)
        sum_col = hist.sum(dim=0)
        denom = sum_row + sum_col - diag
        result = diag / denom
        result[sum_row == 0] = float('nan')
        return result

    def compute_mIoU(self) -> float:
        """Compute mIoU from the accumulated confusion matrix.

        Returns
        -------
        float
            mIoU (0-100) averaged over the first ``num_classes-1`` classes.
        """
        iu = self.per_class_iu(self._hist)          # torch Tensor
        vals = iu[:self.num_classes - 1]
        mean = vals[~vals.isnan()].mean().item()
        return round(mean * 100, 2)

    # ---- logging ----
    def log(self, results: Dict[str, Any]) -> None:
        """Log metrics.

        The base implementation is intentionally a **no-op** — it knows
        nothing about how a subclass structures or presents its output.
        Subclasses are expected to override this method.
        """
        return


# ===================================================================
#  Unified mIoU Metric
# ===================================================================

class Metric_mIoU(BaseMetric):
    """Unified occupancy mIoU evaluator (torch-based).

    In occupancy benchmarks the **last class** (e.g. class 17 in Occ3D,
    class 16 in OpenOccupancy) represents *free / empty* space and is
    **excluded from mIoU**.  When ``compute_binary_iou`` is enabled the
    binary IoU of *non-free* (occupied) voxels is also reported as
    ``'IoU'``.

    All accumulation and index arithmetic run on the configured ``device``
    so evaluation can happen on GPU without host <-> device copies.

    Parameters
    ----------
    num_classes : int
        Total number of classes *including* the free/empty class.
    class_names : list of str
    device : str
    compute_binary_iou : bool
        Additionally compute binary IoU (non-free vs free).
    mask_mode : Optional[str]
        ``'image'`` / ``'lidar'`` / ``None``
    free_label : Optional[int]
        Class index of the *free / empty* voxel in the original label
        space.  Defaults to ``num_classes - 1``.  Must be set explicitly
        when ``num_classes == 2`` because the ``free`` label is still
        e.g. 17 in the underlying Occ3D label space.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: List[str],
        device: str = 'cpu',
        compute_binary_iou: bool = True,
        mask_mode: Optional[str] = None,
        free_label: Optional[int] = None,
    ):
        super().__init__(
            num_classes=num_classes,
            class_names=class_names,
            device=device,
        )
        self.compute_binary_iou = compute_binary_iou
        self.mask_mode = mask_mode

        # free / empty class index in the *original* label space.
        # For Occ3D-18 this is 17; for OpenOccupancy this is 16.
        self.free_label: int = (
            free_label if free_label is not None else num_classes - 1
        )

        self._bin_hist: Optional[torch.Tensor] = None
        if self.compute_binary_iou:
            self._bin_hist = torch.zeros(2, 2, dtype=torch.float64, device=device)

    # ---- device management (handle _bin_hist too) ----
    def to(self, device: str) -> 'Metric_mIoU':
        super().to(device)
        if self._bin_hist is not None:
            self._bin_hist = self._bin_hist.to(device)
        return self

    def reset(self) -> None:
        super().reset()
        if self.compute_binary_iou:
            self._bin_hist = torch.zeros(2, 2, dtype=torch.float64, device=self.device)

    # ---- backward-compatible add_batch alias ----
    def add_batch(
        self,
        semantics_pred,
        semantics_gt,
        *args,
    ) -> None:
        """Backward-compatible wrapper around ``update``.

        Handles two legacy calling conventions:

        * Occ3D  – ``add_batch(pred, gt, mask_lidar, mask_camera)``
        * Occupancy – ``add_batch(pred, gt, mask)``
        """
        if len(args) == 2:
            return self.update(semantics_pred, semantics_gt, mask_lidar=args[0], mask_camera=args[1])
        elif len(args) == 1:
            return self.update(semantics_pred, semantics_gt, mask=args[0])
        else:
            return self.update(semantics_pred, semantics_gt)

    # ---- accumulate ----
    def update(
        self,
        pred,
        gt,
        mask=None,
        mask_lidar=None,
        mask_camera=None,
    ) -> None:
        """Accumulate one sample.

        Masking order (first match wins):
        1. ``mask_mode='image'`` → ``mask_camera``
        2. ``mask_mode='lidar'`` → ``mask_lidar``
        3. explicit ``mask`` kwarg

        Parameters
        ----------
        pred : np.ndarray or torch.Tensor
        gt : np.ndarray or torch.Tensor
        mask : np.ndarray or torch.Tensor, optional
        mask_lidar : np.ndarray or torch.Tensor, optional
        mask_camera : np.ndarray or torch.Tensor, optional
        """
        self.cnt += 1

        # --- convert to torch tensors on the configured device ---
        pred = self._as_tensor(pred, device=self.device)
        gt = self._as_tensor(gt, device=self.device)

        # --- apply mask ---
        if self.mask_mode == 'image' and mask_camera is not None:
            mask_camera = self._as_tensor(mask_camera, device=self.device).bool()
            pred = pred[mask_camera]
            gt = gt[mask_camera]
        elif self.mask_mode == 'lidar' and mask_lidar is not None:
            mask_lidar = self._as_tensor(mask_lidar, device=self.device).bool()
            pred = pred[mask_lidar]
            gt = gt[mask_lidar]
        elif mask is not None:
            mask = self._as_tensor(mask, device=self.device).bool()
            pred = pred[mask]
            gt = gt[mask]

        # --- 2-class special case: map free_label → 1, others → 0 ---
        if self.num_classes == 2:
            pred = pred.clone()
            gt = gt.clone()
            fl = self.free_label
            pred[pred < fl] = 0
            pred[pred == fl] = 1
            gt[gt < fl] = 0
            gt[gt == fl] = 1

        pred_flat = self._to_1d_long(pred)
        gt_flat = self._to_1d_long(gt)

        # --- multi-class histogram ---
        self._hist += self.hist_info(self.num_classes, pred_flat, gt_flat)[0]

        # --- binary IoU histogram ---
        if self.compute_binary_iou:
            bin_pred = pred_flat.clone()
            bin_gt = gt_flat.clone()
            fl = self.free_label
            # free/empty → binary 1, everything else → 0
            bin_pred[bin_pred < fl] = 0
            bin_pred[bin_pred == fl] = 1
            bin_gt[bin_gt < fl] = 0
            bin_gt[bin_gt == fl] = 1
            self._bin_hist += self.hist_info(2, bin_pred, bin_gt)[0]

    # ---- finalise ----
    def compute(self) -> Dict[str, Any]:
        """Return ``{'mIoU': float}`` (plus ``{'IoU': float}`` if enabled).

        mIoU is averaged over the first ``num_classes-1`` classes
        (i.e. excluding the free/empty class).  No logging is performed;
        call :meth:`log` separately when output is desired.
        """
        iu = self.per_class_iu(self._hist)

        # mIoU over *non-free* classes (exclude the last / free class)
        vals = iu[:self.num_classes - 1]
        result = {
            'mIoU': round(vals[~vals.isnan()].mean().item() * 100, 2),
        }

        if self.compute_binary_iou:
            # class 0 = non-free (occupied), class 1 = free (empty)
            iou_arr = self.per_class_iu(self._bin_hist)
            result['IoU'] = round(iou_arr[0].item() * 100, 2)

        return result

    # ---- logging ----
    def _log_per_class_iu(self, iu: torch.Tensor) -> None:
        """Log per-class IoU to the default logger."""
        iu_cpu = iu.cpu()
        logging.info(f'===> per class IoU of {self.cnt} samples:')
        for ind_class in range(self.num_classes - 1):
            logging.info(
                f'===> {self.class_names[ind_class]} - IoU = '
                + str(round(iu_cpu[ind_class].item() * 100, 2))
            )

    def log(self, results: Dict[str, Any]) -> None:
        """Log per-class IoU and scalar metric summary (mIoU / IoU).

        Parameters
        ----------
        results : dict
            The dictionary returned by :meth:`compute`.
        """
        iu = self.per_class_iu(self._hist)
        self._log_per_class_iu(iu)
        logging.info(f'===> mIoU of {self.cnt} samples: {results["mIoU"]}')
        if 'IoU' in results:
            logging.info(f'===> IoU of {self.cnt} samples: {results["IoU"]}')

    # ---- backward-compatible count_miou() ----
    def count_miou(self) -> Union[float, tuple]:
        result = self.compute()
        if self.compute_binary_iou:
            return result['mIoU'], result['IoU']
        return result['mIoU']


# ===================================================================
#  Backward-compatible aliases
# ===================================================================

def Metric_mIoU_Occ3D(
    save_dir: str = '.',
    num_classes: int = 18,
    use_lidar_mask: bool = False,
    use_image_mask: bool = False,
    **kwargs,
) -> Metric_mIoU:
    """Backward-compatible constructor for the old Occ3D metric.

    Occ3D has 18 classes: class 17 = free/empty, which is excluded
    from mIoU.  When ``num_classes=2`` the ``free_label`` must be
    kept at 17 (the real label in the original 18-class space).
    """
    if use_image_mask and use_lidar_mask:
        raise ValueError('use_image_mask and use_lidar_mask are mutually exclusive')

    mask_mode = None
    if use_image_mask:
        mask_mode = 'image'
    elif use_lidar_mask:
        mask_mode = 'lidar'

    class_names = OCC3D_CLASS_NAMES_18 if num_classes == 18 else OCC3D_CLASS_NAMES_2

    return Metric_mIoU(
        num_classes=num_classes,
        class_names=class_names,
        mask_mode=mask_mode,
        compute_binary_iou=True,
        free_label=17,          # Occ3D: class 17 = free / empty
        **kwargs,
    )


def Metric_mIoU_Occupancy(**kwargs) -> Metric_mIoU:
    """Backward-compatible constructor for the old Occupancy metric.

    OpenOccupancy has 17 classes: class 16 = free/empty, excluded
    from mIoU.  Binary IoU (non-free) is also reported.
    """
    return Metric_mIoU(
        num_classes=len(OPENOCC_CLASS_NAMES),
        class_names=OPENOCC_CLASS_NAMES,
        compute_binary_iou=True,
        free_label=len(OPENOCC_CLASS_NAMES) - 1,  # class 16 = free / empty
        **kwargs,
    )