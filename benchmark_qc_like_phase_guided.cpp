#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Clock = std::chrono::high_resolution_clock;

static inline uint8_t keystream_byte(uint64_t key, uint64_t idx) {
    uint64_t x = (key * 0x9E3779B185EBCA87ULL + (idx + 1ULL) * 0xC2B2AE3D27D4EB4FULL);
    x ^= (x >> 33);
    x *= 0xFF51AFD7ED558CCDULL;
    x ^= (x >> 33);
    return static_cast<uint8_t>(x & 0xFFULL);
}

static std::vector<uint8_t> encrypt(const std::vector<uint8_t>& pt, uint64_t key) {
    std::vector<uint8_t> ct(pt.size());
    for (size_t i = 0; i < pt.size(); ++i) {
        ct[i] = static_cast<uint8_t>(pt[i] ^ keystream_byte(key, i));
    }
    return ct;
}

static bool key_matches(const std::vector<uint8_t>& ct, const std::vector<uint8_t>& known, uint64_t key) {
    for (size_t i = 0; i < known.size(); ++i) {
        if ((ct[i] ^ keystream_byte(key, i)) != known[i]) return false;
    }
    return true;
}

static std::pair<uint64_t, uint64_t> brute_force(const std::vector<uint8_t>& ct, const std::vector<uint8_t>& known, int bits) {
    const uint64_t keyspace = 1ULL << bits;
    uint64_t attempts = 0;
    for (uint64_t k = 0; k < keyspace; ++k) {
        ++attempts;
        if (key_matches(ct, known, k)) return {k, attempts};
    }
    return {UINT64_MAX, attempts};
}

static double phase_guided_score(uint64_t key, const std::vector<uint8_t>& ct, const std::vector<uint8_t>& known, int precheck_len) {
    double phi_a = (static_cast<double>(key & 0xFFULL) / 255.0) * (2.0 * M_PI);
    double phi_b = (static_cast<double>((key >> 8) & 0xFFULL) / 255.0) * (2.0 * M_PI);

    constexpr double coupling = 1.9;
    constexpr double leak = 0.02;
    constexpr double dt = 0.015;
    constexpr double anchor_hz = 200.0;
    constexpr double anchor_amp = 0.4;

    double err_acc = 0.0;
    const int n = std::min<int>(precheck_len, static_cast<int>(known.size()));
    for (int i = 0; i < n; ++i) {
        uint8_t pred = static_cast<uint8_t>(ct[i] ^ keystream_byte(key, static_cast<uint64_t>(i)));
        err_acc += std::abs(static_cast<int>(pred) - static_cast<int>(known[i])) / 255.0;
    }
    const double err = err_acc / std::max(1, n);

    double t = 0.0;
    for (int i = 0; i < 8; ++i) {
        const double d = std::sin(phi_b - phi_a);
        const double drive = (1.0 - err) * 1.5;
        const double dphi_a = coupling * d + anchor_amp * std::sin(2.0 * M_PI * anchor_hz * t - phi_a) - leak * phi_a;
        const double dphi_b = -coupling * d + drive * std::cos(phi_b) + anchor_amp * std::sin(2.0 * M_PI * anchor_hz * t - phi_b) - leak * phi_b;
        phi_a = std::fmod(phi_a + dt * dphi_a, 2.0 * M_PI);
        phi_b = std::fmod(phi_b + dt * dphi_b, 2.0 * M_PI);
        if (phi_a < 0) phi_a += 2.0 * M_PI;
        if (phi_b < 0) phi_b += 2.0 * M_PI;
        t += dt;
    }

    const double coherence = std::abs(std::cos(phi_a - phi_b));
    return err + 0.15 * coherence;
}

static std::pair<uint64_t, uint64_t> phase_guided_search(const std::vector<uint8_t>& ct, const std::vector<uint8_t>& known, int bits, int precheck_len) {
    const uint64_t keyspace = 1ULL << bits;
    std::vector<std::pair<double, uint64_t>> scored(keyspace);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t k = 0; k < static_cast<int64_t>(keyspace); ++k) {
        scored[static_cast<size_t>(k)] = {phase_guided_score(static_cast<uint64_t>(k), ct, known, precheck_len), static_cast<uint64_t>(k)};
    }

    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    uint64_t attempts = 0;
    for (const auto& sk : scored) {
        ++attempts;
        if (key_matches(ct, known, sk.second)) return {sk.second, attempts};
    }
    return {UINT64_MAX, attempts};
}

static std::vector<uint64_t> make_fixed_keys(uint64_t keyspace, int reps) {
    std::vector<double> qs;
    if (reps <= 1) qs = {0.5};
    else if (reps == 2) qs = {0.30, 0.70};
    else if (reps == 3) qs = {0.20, 0.50, 0.80};
    else {
        qs.resize(static_cast<size_t>(reps));
        for (int i = 0; i < reps; ++i) qs[static_cast<size_t>(i)] = static_cast<double>(i + 1) / static_cast<double>(reps + 1);
    }
    std::vector<uint64_t> keys;
    keys.reserve(qs.size());
    for (double q : qs) {
        uint64_t k = static_cast<uint64_t>(q * static_cast<double>(keyspace));
        if (k >= keyspace) k = keyspace - 1;
        keys.push_back(k);
    }
    return keys;
}

static int reps_for_bits(int bits) {
    if (bits <= 10) return 6;
    if (bits <= 12) return 4;
    if (bits <= 14) return 3;
    return 2;
}

struct Row {
    int bits = 0;
    uint64_t keyspace = 0;
    int reps = 0;
    std::vector<uint64_t> keys_tested;
    double bf_attempts = 0.0;
    double pg_attempts = 0.0;
    double attempt_gain = 0.0;
    double bf_time = 0.0;
    double pg_time = 0.0;
    double time_gain = 0.0;
    double grover = 0.0;
};

static std::string join_keys(const std::vector<uint64_t>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ", ";
        oss << v[i];
    }
    oss << "]";
    return oss.str();
}

int main(int argc, char** argv) {
    std::vector<int> bits = {8, 10, 12, 14, 16};
    int known_len = 10;
    int precheck_len = 4;
    std::string out_json = "reports/qc_like_phase_guided_benchmark_cpp.json";
    std::string out_md = "reports/qc_like_phase_guided_benchmark_cpp.md";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--known-len" && i + 1 < argc) known_len = std::stoi(argv[++i]);
        else if (a == "--precheck-len" && i + 1 < argc) precheck_len = std::stoi(argv[++i]);
        else if (a == "--out-json" && i + 1 < argc) out_json = argv[++i];
        else if (a == "--out-md" && i + 1 < argc) out_md = argv[++i];
    }

    const std::string base = "REZON_PHASE_GUIDED_DECRYPT_BENCH_";
    std::vector<uint8_t> pt;
    while (pt.size() < 96) pt.insert(pt.end(), base.begin(), base.end());
    pt.resize(96);
    std::vector<uint8_t> known(pt.begin(), pt.begin() + known_len);

    std::vector<Row> rows;
    for (int b : bits) {
        Row r;
        r.bits = b;
        r.keyspace = 1ULL << b;
        r.reps = reps_for_bits(b);
        r.keys_tested = make_fixed_keys(r.keyspace, r.reps);
        std::vector<double> bf_a, pg_a, bf_t, pg_t;
        for (uint64_t key : r.keys_tested) {
            auto ct = encrypt(pt, key);

            auto t0 = Clock::now();
            auto [bf_key, bf_attempts] = brute_force(ct, known, b);
            auto t1 = Clock::now();
            auto [pg_key, pg_attempts] = phase_guided_search(ct, known, b, precheck_len);
            auto t2 = Clock::now();

            if (bf_key != key || pg_key != key) {
                std::cerr << "Key recovery mismatch at bits=" << b << " key=" << key << "\n";
                return 2;
            }

            bf_a.push_back(static_cast<double>(bf_attempts));
            pg_a.push_back(static_cast<double>(pg_attempts));
            bf_t.push_back(std::chrono::duration<double>(t1 - t0).count());
            pg_t.push_back(std::chrono::duration<double>(t2 - t1).count());
        }
        auto mean = [](const std::vector<double>& x) { return std::accumulate(x.begin(), x.end(), 0.0) / std::max<size_t>(1, x.size()); };
        r.bf_attempts = mean(bf_a);
        r.pg_attempts = mean(pg_a);
        r.bf_time = mean(bf_t);
        r.pg_time = mean(pg_t);
        r.attempt_gain = r.bf_attempts / std::max(1e-12, r.pg_attempts);
        r.time_gain = r.bf_time / std::max(1e-12, r.pg_time);
        r.grover = std::sqrt(static_cast<double>(r.keyspace));
        rows.push_back(r);
    }

    {
        std::ofstream jf(out_json);
        jf << "{\n";
        jf << "  \"config\": {\n";
        jf << "    \"bits\": [8, 10, 12, 14, 16],\n";
        jf << "    \"known_len\": " << known_len << ",\n";
        jf << "    \"precheck_len\": " << precheck_len << ",\n";
#ifdef _OPENMP
        jf << "    \"openmp\": true,\n";
        jf << "    \"threads\": " << omp_get_max_threads() << "\n";
#else
        jf << "    \"openmp\": false\n";
#endif
        jf << "  },\n";
        jf << "  \"rows\": [\n";
        for (size_t i = 0; i < rows.size(); ++i) {
            const auto& r = rows[i];
            jf << "    {\n";
            jf << "      \"bits\": " << r.bits << ",\n";
            jf << "      \"keyspace\": " << r.keyspace << ",\n";
            jf << "      \"reps\": " << r.reps << ",\n";
            jf << "      \"keys_tested\": " << join_keys(r.keys_tested) << ",\n";
            jf << "      \"bruteforce_avg_attempts\": " << std::fixed << std::setprecision(6) << r.bf_attempts << ",\n";
            jf << "      \"phase_guided_avg_attempts\": " << r.pg_attempts << ",\n";
            jf << "      \"attempt_gain_x\": " << r.attempt_gain << ",\n";
            jf << "      \"bruteforce_avg_time_s\": " << r.bf_time << ",\n";
            jf << "      \"phase_guided_avg_time_s\": " << r.pg_time << ",\n";
            jf << "      \"time_gain_x\": " << r.time_gain << ",\n";
            jf << "      \"grover_ideal_queries\": " << r.grover << "\n";
            jf << "    }" << (i + 1 < rows.size() ? "," : "") << "\n";
        }
        jf << "  ]\n";
        jf << "}\n";
    }

    {
        std::ofstream mf(out_md);
        mf << "# Phase-Guided Decryption Benchmark (C++)\n\n";
        mf << "| bits | keyspace | bf avg attempts | phase-guided avg attempts | attempts gain | bf avg time (s) | pg avg time (s) | time gain | Grover ref |\n";
        mf << "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n";
        for (const auto& r : rows) {
            mf << "| " << r.bits << " | " << r.keyspace << " | "
               << std::fixed << std::setprecision(2) << r.bf_attempts << " | " << r.pg_attempts << " | "
               << std::setprecision(3) << r.attempt_gain << " | "
               << std::setprecision(6) << r.bf_time << " | " << r.pg_time << " | "
               << std::setprecision(3) << r.time_gain << " | "
               << std::setprecision(2) << r.grover << " |\n";
        }
    }

    std::cout << "saved: " << out_json << "\n";
    std::cout << "saved: " << out_md << "\n";
    return 0;
}
