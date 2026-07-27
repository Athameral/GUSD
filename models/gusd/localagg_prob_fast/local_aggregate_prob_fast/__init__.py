#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch.nn as nn
import torch
import torch.nn.functional as F
from . import _C


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pts,
        points_int,
        means3D,
        means3D_int,
        opas,
        semantics,
        radii,
        cov3D,
        H, W, D
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            pts,
            points_int,
            means3D,
            means3D_int,
            opas,
            semantics,
            radii,
            cov3D,
            H, W, D
        )
        # Invoke C++/CUDA rasterizer
        num_rendered, logits, bin_logits, density, probability, geomBuffer, binningBuffer, imgBuffer = _C.local_aggregate(*args) # todo
        
        # Keep relevant tensors for backward
        ctx.num_rendered = num_rendered
        ctx.H = H
        ctx.W = W
        ctx.D = D
        ctx.save_for_backward(
            geomBuffer, 
            binningBuffer, 
            imgBuffer, 
            means3D,
            pts,
            points_int,
            cov3D,
            opas,
            semantics,
            logits,
            bin_logits,
            density,
            probability
        )
        return logits, bin_logits, density

    @staticmethod # todo
    def backward(ctx, logits_grad, bin_logits_grad, density_grad):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        H = ctx.H
        W = ctx.W
        D = ctx.D
        geomBuffer, binningBuffer, imgBuffer, means3D, pts, points_int, cov3D, opas, semantics, logits, bin_logits, density, probability = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (
            geomBuffer,
            binningBuffer,
            imgBuffer,
            H, W, D,
            num_rendered,
            means3D,
            pts,
            points_int,
            cov3D,
            opas,
            semantics,
            logits,
            bin_logits,
            density,
            probability,
            logits_grad,
            bin_logits_grad,
            density_grad)

        # Compute gradients for relevant tensors by invoking backward method
        means3D_grad, opas_grad, semantics_grad, cov3D_grad = _C.local_aggregate_backward(*args)

        grads = (
            None,
            None,
            means3D_grad,
            None,
            opas_grad,
            semantics_grad,
            None,
            cov3D_grad,
            None, None, None
        )

        return grads

class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, radii_min=1,
                 chunk_size=None, chunk_num=None, chunk_margin=5):
        """Spatial Gaussian aggregator with optional chunked rendering.

        Args:
            chunk_size:  (h, w, d) voxels per chunk (mutually exclusive with chunk_num).
            chunk_num:   (h, w, d) number of chunks per axis; overrides chunk_size.
            chunk_margin: extra voxels around each chunk to include Gaussians from.
            If both are None, no chunking (original single-pass behavior).
        """
        super().__init__()
        self.scale_multiplier = scale_multiplier
        self.H = H
        self.W = W
        self.D = D
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        self.grid_size = grid_size
        self.radii_min = radii_min

        if chunk_num is not None:
            self.chunk_size = (
                (H + chunk_num[0] - 1) // chunk_num[0],
                (W + chunk_num[1] - 1) // chunk_num[1],
                (D + chunk_num[2] - 1) // chunk_num[2],
            )
        else:
            self.chunk_size = chunk_size  # None or (h, w, d) tuple
        self.chunk_margin = chunk_margin

    def forward(self, pts, means3D, opas, semantics, scales, cov3D):
        assert pts.shape[0] == 1
        pts = pts.squeeze(0)
        assert not pts.requires_grad
        means3D = means3D.squeeze(0)
        opas = opas.squeeze(0)
        semantics = semantics.squeeze(0)
        scales_det = scales.detach().squeeze(0)
        cov3D = cov3D.squeeze(0)

        points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int)
        assert points_int.min() >= 0 and points_int[:, 0].max() < self.H \
            and points_int[:, 1].max() < self.W and points_int[:, 2].max() < self.D
        means3D_int = ((means3D.detach() - self.pc_min) / self.grid_size).to(torch.int)
        assert means3D_int.min() >= 0
        assert means3D_int[:, 0].max() < self.H
        assert means3D_int[:, 1].max() < self.W
        assert means3D_int[:, 2].max() < self.D

        # --- Fast path: no chunking ---
        if self.chunk_size is None:
            return self._render_single(
                pts, points_int, means3D, means3D_int,
                opas, semantics, scales_det, cov3D)

        # --- Chunked rendering ---
        ch, cw, cd = self.chunk_size
        margin = self.chunk_margin
        N_total = pts.shape[0]
        C = semantics.shape[-1]
        device = pts.device

        out_logits = torch.zeros(N_total, C, device=device)
        out_bin_logits = torch.zeros(N_total, device=device)
        out_density = torch.zeros(N_total, device=device)

        for h0 in range(0, self.H, ch):
            h1 = min(h0 + ch, self.H)
            for w0 in range(0, self.W, cw):
                w1 = min(w0 + cw, self.W)
                for d0 in range(0, self.D, cd):
                    d1 = min(d0 + cd, self.D)

                    # Voxels strictly inside this chunk (non-overlapping across chunks)
                    vx_mask = (
                        (points_int[:, 0] >= h0) & (points_int[:, 0] < h1) &
                        (points_int[:, 1] >= w0) & (points_int[:, 1] < w1) &
                        (points_int[:, 2] >= d0) & (points_int[:, 2] < d1)
                    )
                    if not vx_mask.any():
                        continue

                    vx_indices = torch.where(vx_mask)[0]

                    # Gaussians that may influence this chunk (chunk + margin)
                    gs_h_lo = max(0, h0 - margin)
                    gs_h_hi = min(self.H, h1 + margin)
                    gs_w_lo = max(0, w0 - margin)
                    gs_w_hi = min(self.W, w1 + margin)
                    gs_d_lo = max(0, d0 - margin)
                    gs_d_hi = min(self.D, d1 + margin)
                    gs_mask = (
                        (means3D_int[:, 0] >= gs_h_lo) & (means3D_int[:, 0] < gs_h_hi) &
                        (means3D_int[:, 1] >= gs_w_lo) & (means3D_int[:, 1] < gs_w_hi) &
                        (means3D_int[:, 2] >= gs_d_lo) & (means3D_int[:, 2] < gs_d_hi)
                    )

                    if not gs_mask.any():
                        # No Gaussian covers this chunk → uniform fallback
                        out_logits[vx_indices, :C - 1] = 1.0 / (C - 1)
                        continue

                    # Shift integer coords to sub-grid to shrink CUDA ``ranges`` buffer.
                    sub_h = gs_h_hi - gs_h_lo
                    sub_w = gs_w_hi - gs_w_lo
                    sub_d = gs_d_hi - gs_d_lo
                    shift = torch.tensor([gs_h_lo, gs_w_lo, gs_d_lo],
                                         dtype=torch.int, device=device)

                    sub_pts = pts[vx_mask]
                    sub_pts_int = points_int[vx_mask] - shift
                    sub_means = means3D[gs_mask]
                    sub_means_int = means3D_int[gs_mask] - shift
                    sub_opas = opas[gs_mask]
                    sub_sem = semantics[gs_mask]
                    sub_scales = scales_det[gs_mask]
                    sub_cov = cov3D[gs_mask]

                    # --- Auto-split Gaussians if radix sort would OOM ---
                    # Estimate num_rendered (Gaussian-tile pairs) for the radix sort.
                    # If > MAX_RENDERED, split Gaussians into sub-batches and combine
                    # via alpha compositing: C=C₁+bin₁·C₂, bin=bin₁·bin₂.
                    est = self._estimate_tiles_touched(
                        sub_scales, sub_means_int, sub_h, sub_w, sub_d)
                    gs_indices = torch.where(gs_mask)[0]
                    MAX_RENDERED = 20_000_000  # ~400 MB safe upper bound

                    if est <= MAX_RENDERED or len(gs_indices) <= 1:
                        # Single-pass
                        sub_logits, sub_bin, sub_density = self._render_single(
                            sub_pts, sub_pts_int,
                            sub_means, sub_means_int,
                            sub_opas, sub_sem, sub_scales, sub_cov,
                            grid_H=sub_h, grid_W=sub_w, grid_D=sub_d,
                        )
                        out_logits[vx_indices] = sub_logits
                        out_bin_logits[vx_indices] = sub_bin
                        out_density[vx_indices] = sub_density
                    else:
                        # Multi-pass: split Gaussians, composite alpha-blended results
                        gs_per_batch = max(1, int(len(gs_indices) * MAX_RENDERED / est))
                        n_vx = vx_indices.shape[0]
                        cum_logits = torch.zeros(n_vx, C, device=device)
                        cum_bin = torch.ones(n_vx, device=device)
                        cum_density = torch.zeros(n_vx, device=device)

                        for gs_start in range(0, len(gs_indices), gs_per_batch):
                            gs_end = min(gs_start + gs_per_batch, len(gs_indices))
                            gs_local = torch.zeros(len(gs_mask), dtype=torch.bool, device=device)
                            gs_local[gs_indices[gs_start:gs_end]] = True

                            pass_logits, pass_bin, pass_density = self._render_single(
                                sub_pts, sub_pts_int,
                                sub_means[gs_local[gs_mask]], sub_means_int[gs_local[gs_mask]],
                                sub_opas[gs_local[gs_mask]], sub_sem[gs_local[gs_mask]],
                                sub_scales[gs_local[gs_mask]], sub_cov[gs_local[gs_mask]],
                                grid_H=sub_h, grid_W=sub_w, grid_D=sub_d,
                            )
                            cum_logits += pass_logits * cum_bin.unsqueeze(-1)
                            cum_bin = cum_bin * pass_bin
                            cum_density += pass_density

                        out_logits[vx_indices] = cum_logits
                        out_bin_logits[vx_indices] = cum_bin
                        out_density[vx_indices] = cum_density

        return out_logits, out_bin_logits, out_density

    def _estimate_tiles_touched(self, scales, means_int, H, W, D):
        """Estimate the number of (Gaussian, tile) pairs for the radix sort.

        This is used to decide whether to split Gaussians into sub-batches
        to avoid ``cub::DeviceRadixSort`` OOM.
        """
        radii = torch.ceil(scales * self.scale_multiplier / self.grid_size).to(torch.int)
        radii = radii.clamp(min=self.radii_min)
        # Tiles covered by each Gaussian ≈ (2r+1)³, clipped to grid bounds
        grid = torch.tensor([H, W, D], device=scales.device)
        covered = (2 * radii + 1).clamp(max=grid).prod(dim=-1).float()
        return covered.sum().item()

    def _render_single(self, pts, points_int, means3D, means3D_int,
                       opas, semantics, scales, cov3D,
                       grid_H=None, grid_W=None, grid_D=None):
        """Single-pass rendering (no chunking / per-chunk).

        grid_H/W/D: override self.H/W/D for sub-grid rendering.
                    When chunking, pass the sub-grid size that covers only
                    the Gaussians in this chunk.  This drastically reduces
                    the ``ranges`` buffer in CUDA (from H×W×D to sub_H×sub_W×sub_D).
        """
        if grid_H is None:
            grid_H, grid_W, grid_D = self.H, self.W, self.D
        radii = torch.ceil(scales * self.scale_multiplier / self.grid_size).to(torch.int)
        radii = radii.clamp(min=self.radii_min)
        assert radii.min() >= 1
        cov3D_flat = cov3D.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

        logits, bin_logits, density = _LocalAggregate.apply(
            pts, points_int, means3D, means3D_int,
            opas, semantics, radii, cov3D_flat,
            grid_H, grid_W, grid_D,
        )
        return logits, bin_logits, density
