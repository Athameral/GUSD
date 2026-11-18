#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import warnings

import torch
import torch.nn as nn

from . import _C


class _LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pts, points_int, means3D, means3D_int, opas, semantics,
                radii, cov3D, H, W, D, tile_size, render_mode):
        result = _C.local_aggregate(
            pts, points_int, means3D, means3D_int, opas, semantics,
            radii, cov3D, H, W, D, tile_size, render_mode)
        count, logits, bin_logits, density, probability, _, geom, bins, image = result
        ctx.num_rendered = count
        ctx.grid = (H, W, D)
        ctx.tile_size = tile_size
        ctx.save_for_backward(geom, bins, image, means3D, pts, points_int,
                              means3D_int, radii, cov3D, opas, semantics,
                              logits, bin_logits, density, probability)
        return logits, bin_logits, density

    @staticmethod
    def backward(ctx, logits_grad, bin_logits_grad, density_grad):
        (geom, bins, image, means, pts, points_int, means_int, radii, cov,
         opas, semantics, logits, bin_logits, density, probability) = ctx.saved_tensors
        grads = _C.local_aggregate_backward(
            geom, bins, image, *ctx.grid, ctx.num_rendered, means, pts,
            points_int, means.new_empty((0,), dtype=torch.long),
            means.new_empty((0,)),
            means.new_empty((0,), dtype=torch.long),
            means.new_empty((0,), dtype=torch.long),
            means_int, radii, cov, opas, semantics, logits,
            bin_logits, density, probability, logits_grad, bin_logits_grad,
            density_grad, ctx.tile_size)
        means_grad, opas_grad, semantics_grad, cov_grad = grads
        return (None, None, means_grad, None, opas_grad, semantics_grad,
                None, cov_grad, None, None, None, None, None)


class _SparseLocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, voxel_axes, means, means_int, opas, semantics,
                radii, cov, H, W, D, tile_size, num_rendered,
                geom, bins, image):
        logits, bin_logits, density, probability = _C.local_aggregate_render(
            indices, voxel_axes, means, means_int, opas, semantics, radii, cov,
            bins, image, num_rendered, H, W, D, tile_size)
        ctx.num_rendered = num_rendered
        ctx.grid = (H, W, D)
        ctx.tile_size = tile_size
        if any(ctx.needs_input_grad[index] for index in (3, 5, 6, 8)):
            word_ids = torch.div(indices, 32, rounding_mode='floor')
            support_bits = torch.zeros(((H * W * D + 31) // 32,),
                                       device=indices.device, dtype=torch.int32)
            bit_values = torch.bitwise_left_shift(
                torch.ones_like(indices, dtype=torch.int32),
                indices.remainder(32).to(torch.int32))
            support_bits.scatter_add_(0, word_ids, bit_values)
            word_prefix = torch.bincount(
                word_ids, minlength=support_bits.numel()).to(torch.int32).cumsum(
                    0, dtype=torch.int32)
        else:
            support_bits = indices.new_empty((0,))
            word_prefix = indices.new_empty((0,))
        ctx.save_for_backward(geom, bins, image, means, indices, voxel_axes,
                              support_bits, word_prefix,
                              means_int, radii, cov, opas, semantics,
                              logits, bin_logits, density, probability)
        return logits, bin_logits, density

    @staticmethod
    def backward(ctx, logits_grad, bin_logits_grad, density_grad):
        (geom, bins, image, means, indices, voxel_axes, support_bits,
         word_prefix, means_int, radii, cov,
         opas, semantics, logits, bin_logits, density, probability) = ctx.saved_tensors
        means_grad, opas_grad, semantics_grad, cov_grad = _C.local_aggregate_backward(
            geom, bins, image, *ctx.grid, ctx.num_rendered, means,
            means.new_empty((0, 3)), means_int.new_empty((0, 3)),
            indices, voxel_axes, support_bits, word_prefix,
            means_int, radii, cov, opas, semantics, logits,
            bin_logits, density, probability, logits_grad, bin_logits_grad,
            density_grad, ctx.tile_size)
        return (None, None, means_grad, None, opas_grad, semantics_grad,
                None, cov_grad, None, None, None, None, None,
                None, None, None)


class LocalAggregator(nn.Module):
    def __init__(self, scale_multiplier, H, W, D, pc_min, grid_size, radii_min=1,
                 chunk_size=None, chunk_num=None, chunk_margin=None, tile_size=4, render_mode="dense"):
        """Gaussian aggregation with runtime-selectable coarse spatial tiles.

        tile_size=2 prioritizes speed, 4 balances memory and speed, and 8
        prioritizes index memory. Legacy chunk options are accepted, but no
        longer perform spatial splitting or Gaussian-batch compositing.
        """
        super().__init__()
        if isinstance(tile_size, bool) or tile_size not in (2, 4, 8):
            raise ValueError('tile_size must be 2, 4 or 8')
        if chunk_size is not None or chunk_num is not None or chunk_margin is not None:
            warnings.warn('chunk_size/chunk_num/chunk_margin are deprecated and '
                          'ignored; use tile_size=2, 4 or 8 instead.',
                          FutureWarning, stacklevel=2)
        if render_mode not in ("dense", "active_tile", "sparse_voxel"):
            raise ValueError("render_mode must be dense, active_tile or sparse_voxel")
        self.render_mode = render_mode
        self.scale_multiplier = scale_multiplier
        self.H, self.W, self.D = H, W, D
        self.grid_size = grid_size
        self.radii_min = radii_min
        self.tile_size = int(tile_size)
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))

    def forward(self, pts, means3D, opas, semantics, scales, cov3D,
                scene_extent=None):
        if pts is not None:
            assert pts.shape[0] == 1
            assert not pts.requires_grad
            pts = pts.squeeze(0).float().contiguous()
        means = means3D.squeeze(0).float().contiguous()
        opas = opas.squeeze(0).float().contiguous()
        semantics = semantics.squeeze(0).float().contiguous()
        cov = cov3D.squeeze(0).float()
        scales = scales.detach().squeeze(0).float()
        if means.shape[0] == 0:
            link = (means.sum() + opas.sum() + semantics.sum() + cov.sum()) * 0
            count = 0 if self.render_mode == 'sparse_voxel' else pts.shape[0]
            logits = means.new_zeros((count, semantics.shape[-1]))
            if semantics.shape[-1] > 1:
                logits[:, :-1] = 1.0 / (semantics.shape[-1] - 1)
            outputs = (logits + link, means.new_zeros(count) + link, means.new_zeros(count) + link)
            if self.render_mode == 'sparse_voxel':
                return (means.new_empty((0,), dtype=torch.long),) + outputs
            return outputs

        means_int = ((means.detach() - self.pc_min) / self.grid_size).to(torch.int).contiguous()
        grid = means_int.new_tensor([self.H, self.W, self.D])
        assert ((means_int >= 0) & (means_int < grid)).all()
        radii = torch.ceil(scales * self.scale_multiplier / self.grid_size).to(torch.int)
        radii = radii.clamp(min=self.radii_min).contiguous()
        assert radii.min() >= 1
        cov_flat = cov.flatten(1)[:, [0, 4, 8, 1, 5, 2]].contiguous()

        if self.render_mode == 'sparse_voxel':
            empty_pts = means.new_empty((0, 3))
            empty_int = means_int.new_empty((0, 3))
            setup = _C.local_aggregate(
                empty_pts, empty_int, means, means_int, opas, semantics,
                radii, cov_flat, self.H, self.W, self.D, self.tile_size, 2)
            num_rendered, support = setup[0], setup[5]
            geom, bins, image = setup[6], setup[7], setup[8]
            indices = support.nonzero(as_tuple=False).squeeze(1)
            del support, setup
            if scene_extent is None:
                scene_extent = (self.H * self.grid_size,
                                self.W * self.grid_size,
                                self.D * self.grid_size)
            extent = scene_extent.to(device=means.device, dtype=means.dtype) \
                if torch.is_tensor(scene_extent) else means.new_tensor(scene_extent)
            voxel_axes = torch.cat([
                (torch.arange(size, device=means.device, dtype=torch.float32) + 0.5) /
                size * extent[axis] + self.pc_min[0, axis]
                for axis, size in enumerate((self.H, self.W, self.D))
            ])
            outputs = _SparseLocalAggregate.apply(
                indices, voxel_axes, means, means_int, opas,
                semantics, radii, cov_flat, self.H, self.W, self.D,
                self.tile_size, num_rendered, geom, bins, image)
            return (indices,) + outputs

        if pts.shape[0] == 0:
            link = (means.sum() + opas.sum() + semantics.sum() + cov.sum()) * 0
            return (means.new_empty((0, semantics.shape[-1])) + link,
                    means.new_empty((0,)) + link, means.new_empty((0,)) + link)
        points_int = ((pts - self.pc_min) / self.grid_size).to(torch.int).contiguous()
        if self.render_mode == "active_tile":
            if pts.shape[0] != self.H * self.W * self.D:
                raise ValueError("active_tile requires the complete ordered voxel grid")
            linear = torch.arange(pts.shape[0], device=pts.device, dtype=torch.int32)
            ordered = ((points_int[:, 0] == linear // (self.W * self.D)) &
                       (points_int[:, 1] == (linear // self.D) % self.W) &
                       (points_int[:, 2] == linear % self.D))
            if not bool(ordered.all()):
                raise ValueError("active_tile requires voxel coordinates in x/y/z linear order")
        assert ((points_int >= 0) & (points_int < grid)).all()
        return _LocalAggregate.apply(pts, points_int, means, means_int, opas,
                                     semantics, radii, cov_flat, self.H,
                                     self.W, self.D, self.tile_size,
                                     1 if self.render_mode == "active_tile" else 0)
