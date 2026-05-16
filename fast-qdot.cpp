#include "fast-qdot.h"
#include <iostream>

using namespace std;

int main()
{
    cout << "LUT Size Recommender." << endl;

    int vec_len;
    cout << "Please enter your activation vector length: ";
    cin >> vec_len;

    int l1_cache_kb;
    cout << "Please enter your L1 cache size (KB): ";
    cin >> l1_cache_kb;

    int l2_cache_kb;
    cout << "Please enter your L2 cache size (KB): ";
    cin >> l2_cache_kb;

    int quant_type;
    cout << "Please enter quantization type (0 = binary, 1 = ternary): ";
    cin >> quant_type;

    // n = number of possible weight values per element (2 for binary, 3 for ternary)
    int n = quant_type ? 3 : 2;
    long long l1_bytes = (long long)l1_cache_kb * 1024;
    long long l2_bytes = (long long)l2_cache_kb * 1024;

    // At depth d, the LUT has n^d entries per activation vector element.
    // Total LUT size = vec_len * n^d * 2 bytes (16-bit entries).
    // We find the max depth before the next level would overflow the cache.
    auto max_depth_for_cache = [&](long long cache_bytes) {
        int depth = 1;
        long long lut_size = (long long)vec_len * 2; // depth 1: vec_len entries * 2 bytes each
        while (lut_size * n <= cache_bytes / 2) { // divided by two because weight vector must also be in cache for inference to happen
            lut_size *= n;  // each extra depth level multiplies entries by n
            depth++;
        }
        return make_pair(depth, lut_size);
        };

    auto [l1_depth, l1_lut_bytes] = max_depth_for_cache(l1_bytes);
    auto [l2_depth, l2_lut_bytes] = max_depth_for_cache(l2_bytes);

    cout << "\n--- Recommendations ---" << endl;
    cout << "L1 cache (" << l1_cache_kb << " KB):" << endl;
    cout << "  Max LUT depth : " << l1_depth << endl;
    cout << "  LUT size      : " << l1_lut_bytes / 1024.0 << " KB" << endl;

    cout << "L2 cache (" << l2_cache_kb << " KB):" << endl;
    cout << "  Max LUT depth : " << l2_depth << endl;
    cout << "  LUT size      : " << l2_lut_bytes / 1024.0 << " KB" << endl;

    return 0;
}