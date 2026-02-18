#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#if __has_include(<cub/device/device_topk.cuh>)
#define HAS_CUB_TOPK 1
#include <cub/device/device_topk.cuh>
#else
#define HAS_CUB_TOPK 0
#endif

static bool read_scores(const std::string& path, std::vector<uint64_t>& keys, std::vector<float>& scores) {
    std::ifstream in(path);
    if (!in) return false;
    uint64_t k;
    float s;
    while (in >> k >> s) {
        keys.push_back(k);
        scores.push_back(s);
    }
    return !keys.empty();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: topk_cuda <in.txt> <k> <out.txt>\n";
        return 2;
    }

    const std::string in_path = argv[1];
    const size_t k = static_cast<size_t>(std::stoul(argv[2]));
    const std::string out_path = argv[3];

    std::vector<uint64_t> h_keys;
    std::vector<float> h_scores;
    if (!read_scores(in_path, h_keys, h_scores)) {
        std::cerr << "failed to load input\n";
        return 3;
    }
    const size_t n = h_keys.size();
    const size_t kk = std::min(k, n);

    thrust::device_vector<uint64_t> d_keys(h_keys.begin(), h_keys.end());
    thrust::device_vector<float> d_scores(h_scores.begin(), h_scores.end());

#if HAS_CUB_TOPK
    // Keep smallest scores.
    thrust::device_vector<uint64_t> d_out_keys(kk);
    thrust::device_vector<float> d_out_scores(kk);

    size_t temp_bytes = 0;
    cub::DeviceTopK::SortPairs(
        nullptr,
        temp_bytes,
        thrust::raw_pointer_cast(d_scores.data()),
        thrust::raw_pointer_cast(d_out_scores.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        thrust::raw_pointer_cast(d_out_keys.data()),
        static_cast<int>(n),
        static_cast<int>(kk),
        cudaStreamDefault,
        false // descending=false => keep smallest
    );

    thrust::device_vector<unsigned char> d_temp(temp_bytes);
    cub::DeviceTopK::SortPairs(
        thrust::raw_pointer_cast(d_temp.data()),
        temp_bytes,
        thrust::raw_pointer_cast(d_scores.data()),
        thrust::raw_pointer_cast(d_out_scores.data()),
        thrust::raw_pointer_cast(d_keys.data()),
        thrust::raw_pointer_cast(d_out_keys.data()),
        static_cast<int>(n),
        static_cast<int>(kk),
        cudaStreamDefault,
        false
    );

    thrust::host_vector<uint64_t> out_keys = d_out_keys;
    thrust::host_vector<float> out_scores = d_out_scores;
#else
    // Fallback: full sort on GPU then take first K.
    thrust::sort_by_key(d_scores.begin(), d_scores.end(), d_keys.begin());
    thrust::host_vector<uint64_t> out_keys(kk);
    thrust::host_vector<float> out_scores(kk);
    thrust::copy_n(d_keys.begin(), kk, out_keys.begin());
    thrust::copy_n(d_scores.begin(), kk, out_scores.begin());
#endif

    std::ofstream out(out_path);
#if HAS_CUB_TOPK
    for (size_t i = 0; i < kk; ++i) out << out_keys[i] << " " << out_scores[i] << "\n";
#else
    for (size_t i = 0; i < kk; ++i) out << out_keys[i] << " " << out_scores[i] << "\n";
#endif

    std::cout << "wrote " << kk << " rows\n";
#if HAS_CUB_TOPK
    std::cout << "mode: cub::DeviceTopK\n";
#else
    std::cout << "mode: thrust_sort_fallback\n";
#endif
    return 0;
}
