#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;


// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched,
	const int tile_size)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	tiles_touched[idx] = 0;

	uint3 rect_min, rect_max;
	getTileRect(points_xyz + 3 * idx, radii + 3 * idx, rect_min, rect_max, grid, tile_size);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) * (rect_max.z - rect_min.z) == 0)
		return;

	tiles_touched[idx] = (rect_max.z - rect_min.z) * (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, int TILE_SIZE, bool ACTIVE_TILE, bool LINEAR_POINTS>
__global__ void renderCUDA(
	const int N,
	const float* __restrict__ pts,
	const int* __restrict__ points_int,
	const int64_t* __restrict__ linear_indices,
	const float* __restrict__ voxel_axes,
	const dim3 grid,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float* __restrict__ means3D,
	const float* __restrict__ cov3D,
	const float* __restrict__ opas,
	const float* __restrict__ semantic,
	float* __restrict__ out_logits,
	float* __restrict__ out_bin_logits,
	float* __restrict__ out_density,
	float* __restrict__ out_probability,
	const int* __restrict__ means_int,
	const int* __restrict__ radii,
    const uint32_t* __restrict__ active_tiles,
    const uint32_t* __restrict__ active_count)
{
    int idx;
    if (ACTIVE_TILE) {
        if (blockIdx.x >= *active_count) return;
        const int tile = active_tiles[blockIdx.x];
        const int tw = (grid.y + TILE_SIZE - 1) / TILE_SIZE;
        const int td = (grid.z + TILE_SIZE - 1) / TILE_SIZE;
        const int local = threadIdx.x;
        const int x = (tile / (tw * td)) * TILE_SIZE + local / (TILE_SIZE * TILE_SIZE);
        const int y = ((tile / td) % tw) * TILE_SIZE + (local / TILE_SIZE) % TILE_SIZE;
        const int z = (tile % td) * TILE_SIZE + local % TILE_SIZE;
        if (x >= grid.x || y >= grid.y || z >= grid.z) return;
        idx = (x * grid.y + y) * grid.z + z;
    } else {
        idx = cg::this_grid().thread_rank();
    }
	if (idx >= N)
	    return;

	int point_int[3];
	float3 point;
	if (LINEAR_POINTS) {
		const int64_t linear = linear_indices[idx];
		point_int[0] = linear / (int64_t(grid.y) * grid.z);
		point_int[1] = (linear / grid.z) % grid.y;
		point_int[2] = linear % grid.z;
		// Axis LUTs are produced with the exact dense PyTorch expression.
		// Their O(H+W+D) storage avoids both N- and K-sized coordinate tensors
		// while preserving threshold-sensitive coordinates bit for bit.
		point.x = voxel_axes[point_int[0]];
		point.y = voxel_axes[grid.x + point_int[1]];
		point.z = voxel_axes[grid.x + grid.y + point_int[2]];
	} else {
		point_int[0] = points_int[3 * idx];
		point_int[1] = points_int[3 * idx + 1];
		point_int[2] = points_int[3 * idx + 2];
		point = {pts[3 * idx], pts[3 * idx + 1], pts[3 * idx + 2]};
	}
	const int tile_w = (grid.y + TILE_SIZE - 1) / TILE_SIZE;
	const int tile_d = (grid.z + TILE_SIZE - 1) / TILE_SIZE;
	const int voxel_idx = (point_int[0] / TILE_SIZE) * tile_w * tile_d + (point_int[1] / TILE_SIZE) * tile_d + point_int[2] / TILE_SIZE;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[voxel_idx];

	// Initialize helper variables
	float C[CHANNELS] = { 0 };
	float bin_logit = 1.0;
	float density = 0.0;
	float prob_sum = 0.0;

	for (int i = range.x; i < range.y; i++)
	{
		int gs_idx = point_list[i];
		bool inside = true;
		for (int axis = 0; axis < 3; ++axis) {
			const int center = means_int[3 * gs_idx + axis];
			const int radius = radii[3 * gs_idx + axis];
			inside = inside && point_int[axis] >= center - radius && point_int[axis] <= center + radius;
		}
		if (!inside) continue;
		float3 cov1 = { cov3D[gs_idx * 6 + 0], cov3D[gs_idx * 6 + 1], cov3D[gs_idx * 6 + 2] };
		float3 cov2 = { cov3D[gs_idx * 6 + 3], cov3D[gs_idx * 6 + 4], cov3D[gs_idx * 6 + 5] };
		float3 d = { means3D[gs_idx * 3] - point.x, means3D[gs_idx * 3 + 1] - point.y, means3D[gs_idx * 3 + 2] - point.z };
		float power = cov1.x * d.x * d.x + cov1.y * d.y * d.y + cov1.z * d.z * d.z;
		power = -0.5f * power - (cov2.x * d.x * d.y + cov2.y * d.y * d.z + cov2.z * d.x * d.z);
		power = exp(power);
		float deter = cov1.x * cov1.y * cov1.z + 2 * cov2.x * cov2.y * cov2.z - cov1.x * cov2.y * cov2.y - cov1.y * cov2.z * cov2.z - cov1.z * cov2.x * cov2.x;
		float prob = powf(2 * 3.1415926535, -1.5) * powf(deter, 0.5) * power * opas[gs_idx];

		for (int ch = 0; ch < CHANNELS; ch++)
		{
			C[ch] += semantic[CHANNELS * gs_idx + ch] * prob;
		}
		bin_logit = (1 - power) * bin_logit;
		density = power + density;
		prob_sum = prob + prob_sum;
	}

	// Iterate over batches until all done or range is complete
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (prob_sum > 1e-9) {
		for (int ch = 0; ch < CHANNELS; ch++)
			out_logits[idx * CHANNELS + ch] = C[ch] / prob_sum;
	} else {
		for (int ch = 0; ch < CHANNELS - 1; ch++)
			out_logits[idx * CHANNELS + ch] = 1.0 / (CHANNELS - 1);
	}
	out_bin_logits[idx] = 1 - bin_logit;
	out_density[idx] = density;
	out_probability[idx] = prob_sum;
}


void FORWARD::render(
	const int N,
	const float* pts,
	const int* points_int,
	const dim3 grid,
	const uint2* ranges,
	const uint32_t* point_list,
	const float* means3D,
	const float* cov3D,
	const float* opas,
	const float* semantic,
	float* out_logits,
	float* out_bin_logits,
	float* out_density,
	float* out_probability,
	const int* means_int,
	const int* radii,
	const int tile_size, const bool active_tile, const int tile_count,
    const uint32_t* active_tiles, const uint32_t* active_count)
{
	#define LAUNCH_FORWARD(TILE) do { if (active_tile) renderCUDA<NUM_LOGITS_CHANNELS, TILE, true, false> <<< tile_count, TILE * TILE * TILE >>> (N, pts, points_int, nullptr, nullptr, grid, ranges, point_list, means3D, cov3D, opas, semantic, out_logits, out_bin_logits, out_density, out_probability, means_int, radii, active_tiles, active_count); else renderCUDA<NUM_LOGITS_CHANNELS, TILE, false, false> <<< (N + 255) / 256, 256 >>> (N, pts, points_int, nullptr, nullptr, grid, ranges, point_list, means3D, cov3D, opas, semantic, out_logits, out_bin_logits, out_density, out_probability, means_int, radii, active_tiles, active_count); } while (0)
	switch (tile_size) {
		case 2: LAUNCH_FORWARD(2); break;
		case 4: LAUNCH_FORWARD(4); break;
		case 8: LAUNCH_FORWARD(8); break;
	}
	#undef LAUNCH_FORWARD
}

void FORWARD::renderSparse(
	const int N, const int64_t* linear_indices, const float* voxel_axes,
	const dim3 grid, const uint2* ranges,
	const uint32_t* point_list, const float* means3D, const float* cov3D,
	const float* opas, const float* semantic, float* out_logits,
	float* out_bin_logits, float* out_density, float* out_probability,
	const int* means_int, const int* radii, const int tile_size)
{
	#define LAUNCH_SPARSE(TILE) renderCUDA<NUM_LOGITS_CHANNELS, TILE, false, true> <<< (N + 255) / 256, 256 >>> (N, nullptr, nullptr, linear_indices, voxel_axes, grid, ranges, point_list, means3D, cov3D, opas, semantic, out_logits, out_bin_logits, out_density, out_probability, means_int, radii, nullptr, nullptr)
	switch (tile_size) {
		case 2: LAUNCH_SPARSE(2); break;
		case 4: LAUNCH_SPARSE(4); break;
		case 8: LAUNCH_SPARSE(8); break;
	}
	#undef LAUNCH_SPARSE
}

template <int TILE_SIZE>
__global__ void markSupportCUDA(
	const dim3 grid,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const int* __restrict__ means_int,
	const int* __restrict__ radii,
	bool* __restrict__ support_mask,
	const uint32_t* __restrict__ active_tiles,
	const uint32_t* __restrict__ active_count)
{
	constexpr int VOXELS = TILE_SIZE * TILE_SIZE * TILE_SIZE;
	constexpr int WORDS = (VOXELS + 31) / 32;
	__shared__ uint32_t bits[WORDS];
	for (int word = threadIdx.x; word < WORDS; word += blockDim.x)
		bits[word] = 0;
	__syncthreads();

	const int tile_w = (grid.y + TILE_SIZE - 1) / TILE_SIZE;
	const int tile_d = (grid.z + TILE_SIZE - 1) / TILE_SIZE;
	if (blockIdx.x >= *active_count)
		return;
	const int tile = active_tiles[blockIdx.x];
	const uint2 range = ranges[tile];
	const int tx = tile / (tile_w * tile_d);
	const int ty = (tile / tile_d) % tile_w;
	const int tz = tile % tile_d;
	const int base_x = tx * TILE_SIZE;
	const int base_y = ty * TILE_SIZE;
	const int base_z = tz * TILE_SIZE;

	for (int pair = range.x + threadIdx.x; pair < range.y; pair += blockDim.x) {
		const int gs = point_list[pair];
		const int lo_x = max(base_x, means_int[3 * gs] - radii[3 * gs]);
		const int lo_y = max(base_y, means_int[3 * gs + 1] - radii[3 * gs + 1]);
		const int lo_z = max(base_z, means_int[3 * gs + 2] - radii[3 * gs + 2]);
		const int hi_x = min(base_x + TILE_SIZE - 1, min((int)grid.x - 1, means_int[3 * gs] + radii[3 * gs]));
		const int hi_y = min(base_y + TILE_SIZE - 1, min((int)grid.y - 1, means_int[3 * gs + 1] + radii[3 * gs + 1]));
		const int hi_z = min(base_z + TILE_SIZE - 1, min((int)grid.z - 1, means_int[3 * gs + 2] + radii[3 * gs + 2]));
		for (int x = lo_x; x <= hi_x; ++x)
		for (int y = lo_y; y <= hi_y; ++y)
		for (int z = lo_z; z <= hi_z; ++z) {
			const int local = ((x - base_x) * TILE_SIZE + y - base_y) * TILE_SIZE + z - base_z;
			atomicOr(bits + local / 32, 1u << (local % 32));
		}
	}
	__syncthreads();

	for (int local = threadIdx.x; local < VOXELS; local += blockDim.x) {
		if (!(bits[local / 32] & (1u << (local % 32))))
			continue;
		const int x = base_x + local / (TILE_SIZE * TILE_SIZE);
		const int y = base_y + (local / TILE_SIZE) % TILE_SIZE;
		const int z = base_z + local % TILE_SIZE;
		if (x < grid.x && y < grid.y && z < grid.z)
			support_mask[(x * grid.y + y) * grid.z + z] = true;
	}
}

void FORWARD::markSupport(
	const dim3 grid,
	const uint2* ranges,
	const uint32_t* point_list,
	const int* means_int,
	const int* radii,
	bool* support_mask,
	const uint32_t* active_tiles,
	const uint32_t* active_count,
	const int tile_size)
{
	const int tile_count = ((grid.x + tile_size - 1) / tile_size) *
		((grid.y + tile_size - 1) / tile_size) *
		((grid.z + tile_size - 1) / tile_size);
	#define LAUNCH_SUPPORT(TILE) markSupportCUDA<TILE><<<tile_count, 128>>>(grid, ranges, point_list, means_int, radii, support_mask, active_tiles, active_count)
	switch (tile_size) {
		case 2: LAUNCH_SUPPORT(2); break;
		case 4: LAUNCH_SUPPORT(4); break;
		case 8: LAUNCH_SUPPORT(8); break;
	}
	#undef LAUNCH_SUPPORT
}

void FORWARD::preprocess(
	const int P,
	const int* points_xyz,
	const int* radii,
	const dim3 grid,
	uint32_t* tiles_touched,
	const int tile_size)
{
	preprocessCUDA << <(P + 255) / 256, 256 >> > (
		P,
		points_xyz,
		radii,
		grid,
		tiles_touched,
		tile_size
	);
}
