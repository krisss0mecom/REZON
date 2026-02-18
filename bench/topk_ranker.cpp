#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// Minimal C++ top-K ranker (no full sort):
// input: text file with lines "<key> <score>"
// output: top-K lowest scores

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: topk_ranker <in.txt> <k> <out.txt>\n";
        return 2;
    }

    const std::string in_path = argv[1];
    const size_t k = static_cast<size_t>(std::stoul(argv[2]));
    const std::string out_path = argv[3];

    std::ifstream in(in_path);
    if (!in) {
        std::cerr << "cannot open input\n";
        return 3;
    }

    std::vector<std::pair<double, uint64_t>> data;
    data.reserve(1 << 20);

    uint64_t key;
    double score;
    while (in >> key >> score) {
        data.push_back({score, key});
    }

    if (data.empty()) {
        std::cerr << "no data\n";
        return 4;
    }

    const size_t kk = std::min(k, data.size());
    std::nth_element(data.begin(), data.begin() + kk, data.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    std::sort(data.begin(), data.begin() + kk,
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::ofstream out(out_path);
    for (size_t i = 0; i < kk; ++i) {
        out << data[i].second << " " << data[i].first << "\n";
    }

    std::cout << "wrote " << kk << " rows\n";
    return 0;
}
