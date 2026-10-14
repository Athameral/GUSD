# Gaussian Renderer

The renderer uses coarse spatial tiles and a parallel, fixed-order backward
reduction. Set `model.pts_bbox_head.cuda_kwargs.tile_size` to `2`, `4`, or `8`.
The default is `4`; switching sizes does not require recompilation.

- `2`: often faster, but uses more index memory.
- `4`: recommended memory/speed balance.
- `8`: lower index memory, with more candidate checks and potentially slower backward.

Existing configurations without `tile_size` use `4`. Legacy `chunk_size`,
`chunk_num`, and `chunk_margin` are accepted with a `FutureWarning` and ignored.
The old spatial splitting and Gaussian-sub-batch compositing path has been removed;
its compositing did not preserve the probability-mixture equations.

Rebuild the extension after updating the sources:

```bash
cd models/gusd/localagg_prob_fast
python setup.py build_ext --inplace
```

Old `_C` binaries are not compatible with the new backward signature. The public
`LocalAggregator` call and its `pc_min` checkpoint buffer are unchanged.

Forward preserves the original finite support boxes and Gaussian equations.
Backward changes summation order, so it is tolerance-equivalent, not bitwise
identical to the old implementation. Fixed-order reduction is repeatable on the
same device. Numerical covariance/determinant limitations remain; retain the
existing jitter unless separately validated.

Coarse indexing is not an OOM guarantee. Full output grids, upstream gradients,
multiple supervised layers, backbone activations, and optimizer state still use
memory. H20 synthetic stress tests at a 400x400x64 grid, 28,800 Gaussians, and
independent uniformly distributed axis scales in [0.01, 2.4] metres measured
extra single-renderer forward/backward peaks of approximately 20.3 GiB (tile 2),
4.05 GiB (tile 4), and 1.89 GiB (tile 8). These are not full training peaks or
measurements from trained Gaussian distributions.

Queries must use unique voxel indices for backward: the dense voxel-to-point map
stores one point per voxel, as in the original renderer. Empty Gaussian/query
inputs return valid empty/fallback outputs with zero parameter gradients.