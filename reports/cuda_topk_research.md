# CUDA Top-K Research (for REZON phase-guided ranking)

## Goal
Maximize end-to-end throughput for candidate ranking in key search / combinatorial search, while preserving exactness of top-K selection.

## Primary sources reviewed
- NVIDIA CUDA C++ Best Practices Guide:
  - https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- NVIDIA CUDA Programming Guide:
  - https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA CCCL / CUB DeviceTopK API:
  - https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceTopK.html
- NVIDIA CCCL / CUB DeviceRadixSort API:
  - https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceRadixSort.html
- NVIDIA Thrust sorting API:
  - https://nvidia.github.io/cccl/thrust/api/group__sorting_1ga1099d781e06c43805be06a918f7b7499.html
- FAISS GPU (industrial high-performance top-k style retrieval path):
  - https://github.com/facebookresearch/faiss
- GPU-TopK reference implementation:
  - https://github.com/anilshanbhag/gpu-topk

## Practical conclusions
1. **Do not full-sort unless necessary.**
- For top-K, full `sort` is often wasted work.
- Prefer selection primitives (`DeviceTopK`, `nth_element`, block-level selection, radix-select).

2. **Use CUB DeviceTopK if available in your CUDA/CCCL stack.**
- Best direct fit for exact top-K on GPU.
- In this repo, implementation uses `DeviceTopK` when available and fallback otherwise.

3. **Fallback path:**
- If `DeviceTopK` is unavailable (older CUDA/CCCL):
  - Use GPU sort fallback (`thrust::sort_by_key`) first.
  - Next upgrade step: `DeviceRadixSort + truncated merge` or custom selection kernel.

4. **Top-K should be fused with scoring when possible.**
- Current bottleneck is separate `score-all` then `rank-all`.
- Better design:
  - score kernel writes local top-k per block
  - global reduction merges block top-k
- Avoid writing all scores to global memory if not needed.

5. **Two-stage ranking improves throughput.**
- Stage A: cheap coarse score on all keys.
- Stage B: expensive RC score only on shortlist (e.g. top 1-5%).
- This usually dominates any low-level micro-optimization in wall-clock time.

6. **Data layout and transfer discipline matter.**
- Keep ciphertext/prefix/oracle data resident on GPU across iterations.
- Use pinned host buffers for transfers if needed.
- Batch multiple instances to amortize kernel launch overhead.

## REZON-specific recommendation stack (priority order)
1. Keep C++/CUDA path as default for ranking.
2. Switch from full-space exact ranking to **coarse-to-fine top-K**:
   - top-K1 from cheap score
   - re-rank K1 with full RC dynamics
3. Add stream overlap (compute next batch while validating current batch).
4. Tune K values by Pareto search (attempt gain vs runtime).

## Benchmark protocol for CUDA ranking
Track at minimum:
- `rank_time_s`
- `score_time_s`
- `total_time_s`
- `attempts_to_solution`
- `k`, `keyspace`, `batch_size`
- GPU info (`name`, `driver`, `cuda runtime`)

## Current status in this repo
- Added C++ top-K without full sort on CPU (`bench/topk_ranker.cpp`).
- Added CUDA ranking implementation with CUB top-K (if available) and fallback (`bench/topk_cuda.cu`).
- Added wrappers for reproducible smoke runs.

## Claim boundary
This work supports the claim:
- **"quantum-inspired analog search heuristic"**

It does **not** support:
- "QC replacement"
