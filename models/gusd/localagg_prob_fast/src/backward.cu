#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Perform initial steps for each Gaussian prior to rasterization.
__global__ void preprocessCUDA(
	const int N,
	const int* points_xyz,
	const dim3 grid,
	int* voxel2pts)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= N)
		return;

	int voxel_idx = points_xyz[3 * idx] * grid.y * grid.z + points_xyz[3 * idx + 1] * grid.z + points_xyz[3 * idx + 2];
	voxel2pts[voxel_idx] = idx;
}

__global__ void preprocessLinearCUDA(
	const int N,
	const int64_t* linear_indices,
	int* voxel2pts)
{
	const int idx = cg::this_grid().thread_rank();
	if (idx < N)
		voxel2pts[linear_indices[idx]] = idx;
}

__global__ void markSupportBitsCUDA(
	const int N,
	const int64_t* linear_indices,
	uint32_t* support_bits)
{
	const int idx = cg::this_grid().thread_rank();
	if (idx < N) {
		const int64_t linear = linear_indices[idx];
		atomicOr(support_bits + linear / 32, 1u << (linear % 32));
	}
}

__global__ void countSupportBitsCUDA(
	const int word_count,
	const uint32_t* support_bits,
	int* word_counts)
{
	const int idx = cg::this_grid().thread_rank();
	if (idx < word_count)
		word_counts[idx] = __popc(support_bits[idx]);
}


template <uint32_t CHANNELS, int TILE_SIZE, bool LINEAR_POINTS, bool COMPRESSED_MAPPING>
__global__ void renderCUDA(
	const int P,
	const uint32_t* __restrict__ offsets,
	const uint32_t* __restrict__ point_list_keys_unsorted,
	const int* __restrict__ means_int,
	const int* __restrict__ radii,
	const dim3 grid,
	const int* __restrict__ voxel2pts,
	const uint32_t* __restrict__ support_bits,
	const int* __restrict__ word_prefix,
	const float* __restrict__ pts,
	const int64_t* __restrict__ linear_indices,
	const float* __restrict__ voxel_axes,
	const float* __restrict__ means3D,
	const float* __restrict__ cov3D,
	const float* __restrict__ opas,
	const float* __restrict__ semantic,
	const float* __restrict__ logits,
	const float* __restrict__ bin_logits,
	const float* __restrict__ density,
	const float* __restrict__ probability,
	const float* __restrict__ logits_grad,
	const float* __restrict__ bin_logits_grad,
	const float* __restrict__ density_grad,
	float* __restrict__ means3D_grad,
	float* __restrict__ opas_grad,
	float* __restrict__ semantics_grad,
	float* __restrict__ cov3D_grad)
{
	const int idx = blockIdx.x;
	const int lane = threadIdx.x;
	if (idx >= P)
	    return;

	uint32_t start = (idx == 0) ? 0 : offsets[idx - 1];
	uint32_t end = offsets[idx];
	
	const float3 means = {means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2]};
	const float3 cov1 = {cov3D[6 * idx], cov3D[6 * idx + 1], cov3D[6 * idx + 2]};
	const float3 cov2 = {cov3D[6 * idx + 3], cov3D[6 * idx + 4], cov3D[6 * idx + 5]};
	const float opa = opas[idx];
	float sem[CHANNELS] = {0};
	for (int ch = 0; ch < CHANNELS; ch++)
	{
		sem[ch] = semantic[idx * CHANNELS + ch];
	}

	float means_grad[3] = {0};
	float opa_grad = 0;
	float semantic_grad[CHANNELS] = {0};
	float cov_grad[6] = {0};

	const int tile_w = (grid.y + TILE_SIZE - 1) / TILE_SIZE;
	const int tile_d = (grid.z + TILE_SIZE - 1) / TILE_SIZE;
	for (int i = start + lane; i < end; i += blockDim.x)
	{
		int tile_idx = point_list_keys_unsorted[i];
		int tz = tile_idx % tile_d;
		int ty = (tile_idx / tile_d) % tile_w;
		int tx = tile_idx / (tile_w * tile_d);
		for (int lx = 0; lx < TILE_SIZE; ++lx)
		for (int ly = 0; ly < TILE_SIZE; ++ly)
		for (int lz = 0; lz < TILE_SIZE; ++lz)
		{
			int vx = tx * TILE_SIZE + lx;
			int vy = ty * TILE_SIZE + ly;
			int vz = tz * TILE_SIZE + lz;
			if (vx >= grid.x || vy >= grid.y || vz >= grid.z)
				continue;
			if (vx < means_int[3 * idx] - radii[3 * idx] || vx > means_int[3 * idx] + radii[3 * idx] ||
				vy < means_int[3 * idx + 1] - radii[3 * idx + 1] || vy > means_int[3 * idx + 1] + radii[3 * idx + 1] ||
				vz < means_int[3 * idx + 2] - radii[3 * idx + 2] || vz > means_int[3 * idx + 2] + radii[3 * idx + 2])
				continue;
			int voxel_idx = vx * grid.y * grid.z + vy * grid.z + vz;
			int pts_idx;
			if (COMPRESSED_MAPPING) {
				const int word = voxel_idx / 32;
				const int bit = voxel_idx % 32;
				const uint32_t value = support_bits[word];
				if (!(value & (1u << bit))) {
					pts_idx = -1;
				} else {
					const int before = word == 0 ? 0 : word_prefix[word - 1];
					const uint32_t lower = bit == 0 ? 0u : value & ((1u << bit) - 1u);
					pts_idx = before + __popc(lower);
				}
			} else {
				pts_idx = voxel2pts[voxel_idx];
			}
			if (pts_idx >= 0)
		{
			float3 point;
			if (LINEAR_POINTS) {
				const int64_t linear = linear_indices[pts_idx];
				const int px = linear / (int64_t(grid.y) * grid.z);
				const int py = (linear / grid.z) % grid.y;
				const int pz = linear % grid.z;
				point.x = voxel_axes[px];
				point.y = voxel_axes[grid.x + py];
				point.z = voxel_axes[grid.x + grid.y + pz];
			} else {
				point = {pts[pts_idx * 3], pts[pts_idx * 3 + 1], pts[pts_idx * 3 + 2]};
			}
			float3 d = {means.x - point.x, means.y - point.y, means.z - point.z};
			float power = cov1.x * d.x * d.x + cov1.y * d.y * d.y + cov1.z * d.z * d.z;
			power = -0.5f * power - (cov2.x * d.x * d.y + cov2.y * d.y * d.z + cov2.z * d.x * d.z);
			power = exp(power);
			float deter = cov1.x * cov1.y * cov1.z + 2 * cov2.x * cov2.y * cov2.z - cov1.x * cov2.y * cov2.y - cov1.y * cov2.z * cov2.z - cov1.z * cov2.x * cov2.x;
			float prob = powf(2 * 3.1415926535, -1.5) * powf(deter, 0.5) * power;
			float power_grad = 0.;
			float deter_grad = 0.;
			float prob_grad = 0.;
			float prob_sum = probability[pts_idx];

			if (prob_sum > 1e-9) {
				for (int ch = 0; ch < CHANNELS; ch++)
				{
					semantic_grad[ch] += logits_grad[pts_idx * CHANNELS + ch] * prob * opa / prob_sum;
					prob_grad += logits_grad[pts_idx * CHANNELS + ch] * (sem[ch] - logits[pts_idx * CHANNELS + ch]) * opa / prob_sum;
					opa_grad += logits_grad[pts_idx * CHANNELS + ch] * (sem[ch] - logits[pts_idx * CHANNELS + ch]) * prob / prob_sum;
				}
			} 
			power_grad += prob_grad * powf(2 * 3.1415926535, -1.5) * powf(deter, 0.5);
			power_grad += (1 - bin_logits[pts_idx]) / (1 - power + 1e-9) *  bin_logits_grad[pts_idx];
			power_grad += density_grad[pts_idx];
			deter_grad += prob_grad * prob / 2 / deter;

			means_grad[0] -= power_grad * power * (cov1.x * d.x + cov2.x * d.y + cov2.z * d.z);
			means_grad[1] -= power_grad * power * (cov2.x * d.x + cov1.y * d.y + cov2.y * d.z);
			means_grad[2] -= power_grad * power * (cov2.z * d.x + cov2.y * d.y + cov1.z * d.z);

			cov_grad[0] += power_grad * power * (- 0.5 * d.x * d.x) + deter_grad * (cov1.y * cov1.z - cov2.y * cov2.y);
			cov_grad[1] += power_grad * power * (- 0.5 * d.y * d.y) + deter_grad * (cov1.x * cov1.z - cov2.z * cov2.z);
			cov_grad[2] += power_grad * power * (- 0.5 * d.z * d.z) + deter_grad * (cov1.x * cov1.y - cov2.x * cov2.x);
			cov_grad[3] += power_grad * power * (- d.x * d.y) + 2 * deter_grad * (cov2.y * cov2.z - cov1.z * cov2.x);
			cov_grad[4] += power_grad * power * (- d.y * d.z) + 2 * deter_grad * (cov2.x * cov2.z - cov1.x * cov2.y);
			cov_grad[5] += power_grad * power * (- d.x * d.z) + 2 * deter_grad * (cov2.x * cov2.y - cov1.y * cov2.z);
		}
		}
	}

	__shared__ float partial[CHANNELS + 10][128];
	partial[0][lane] = means_grad[0];
	partial[1][lane] = means_grad[1];
	partial[2][lane] = means_grad[2];
	partial[3][lane] = opa_grad;
	for (int ch = 0; ch < CHANNELS; ch++)
	{
		partial[4 + ch][lane] = semantic_grad[ch];
	}
	for (int ch = 0; ch < 6; ch++)
	{
		partial[4 + CHANNELS + ch][lane] = cov_grad[ch];
	}
	__syncthreads();
	for (int stride = 64; stride > 0; stride /= 2) {
		if (lane < stride) {
			for (int ch = 0; ch < CHANNELS + 10; ++ch)
				partial[ch][lane] += partial[ch][lane + stride];
		}
		__syncthreads();
	}
	if (lane == 0) {
		for (int ch = 0; ch < 3; ++ch) means3D_grad[idx * 3 + ch] = partial[ch][0];
		opas_grad[idx] = partial[3][0];
		for (int ch = 0; ch < CHANNELS; ++ch) semantics_grad[idx * CHANNELS + ch] = partial[4 + ch][0];
		for (int ch = 0; ch < 6; ++ch) cov3D_grad[idx * 6 + ch] = partial[4 + CHANNELS + ch][0];
	}
}


void BACKWARD::render(
	const int P,
	const uint32_t* offsets,
	const uint32_t* point_list_keys_unsorted,
	const int* means_int,
	const int* radii,
	const dim3 grid,
	const int* voxel2pts,
	const uint32_t* support_bits,
	const int* word_prefix,
	const float* pts,
	const int64_t* linear_indices,
	const float* voxel_axes,
	const float* means3D,
	const float* cov3D,
	const float* opas,
	const float* semantic,
	const float* logits,
	const float* bin_logits,
	const float* density,
	const float* probability,
	const float* logits_grad,
	const float* bin_logits_grad,
	const float* density_grad,
	float* means3D_grad,
	float* opas_grad,
	float* semantics_grad,
	float* cov3D_grad,
	const int tile_size,
	const bool sparse_points,
	const bool compressed_mapping)
{
	#define LAUNCH_BACKWARD(TILE) do { if (compressed_mapping) renderCUDA<NUM_LOGITS_CHANNELS, TILE, true, true> <<< P, 128 >>> (P, offsets, point_list_keys_unsorted, means_int, radii, grid, nullptr, support_bits, word_prefix, nullptr, linear_indices, voxel_axes, means3D, cov3D, opas, semantic, logits, bin_logits, density, probability, logits_grad, bin_logits_grad, density_grad, means3D_grad, opas_grad, semantics_grad, cov3D_grad); else if (sparse_points) renderCUDA<NUM_LOGITS_CHANNELS, TILE, true, false> <<< P, 128 >>> (P, offsets, point_list_keys_unsorted, means_int, radii, grid, voxel2pts, nullptr, nullptr, nullptr, linear_indices, voxel_axes, means3D, cov3D, opas, semantic, logits, bin_logits, density, probability, logits_grad, bin_logits_grad, density_grad, means3D_grad, opas_grad, semantics_grad, cov3D_grad); else renderCUDA<NUM_LOGITS_CHANNELS, TILE, false, false> <<< P, 128 >>> (P, offsets, point_list_keys_unsorted, means_int, radii, grid, voxel2pts, nullptr, nullptr, pts, nullptr, nullptr, means3D, cov3D, opas, semantic, logits, bin_logits, density, probability, logits_grad, bin_logits_grad, density_grad, means3D_grad, opas_grad, semantics_grad, cov3D_grad); } while (0)
	switch (tile_size) {
		case 2: LAUNCH_BACKWARD(2); break;
		case 4: LAUNCH_BACKWARD(4); break;
		case 8: LAUNCH_BACKWARD(8); break;
	}
	#undef LAUNCH_BACKWARD
}

void BACKWARD::preprocessLinear(
	const int N,
	const int64_t* linear_indices,
	int* voxel2pts)
{
	preprocessLinearCUDA<<<(N + 255) / 256, 256>>>(N, linear_indices, voxel2pts);
}

void BACKWARD::markSupportBits(
	const int N,
	const int64_t* linear_indices,
	uint32_t* support_bits)
{
	markSupportBitsCUDA<<<(N + 255) / 256, 256>>>(N, linear_indices, support_bits);
}

void BACKWARD::countSupportBits(
	const int word_count,
	const uint32_t* support_bits,
	int* word_counts)
{
	countSupportBitsCUDA<<<(word_count + 255) / 256, 256>>>(word_count, support_bits, word_counts);
}

void BACKWARD::preprocess(
	const int N,
	const int* points_xyz,
	const dim3 grid,
	int* voxel2pts)
{
	preprocessCUDA << <(N + 255) / 256, 256 >> > (
		N,
		points_xyz,
		grid,
		voxel2pts
	);
}
