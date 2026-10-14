#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

// Default HPC simulation parameters
constexpr int DEFAULT_SIZE = 2000;
constexpr int DEFAULT_ITERATIONS = 100;

/**
 * Optimized next generation step.
 * Uses flat 1D memory layout (matching CUDA structure) and OpenMP multithreading.
 */
void nextGenerationOptimized(const int* __restrict__ current, int* __restrict__ next, int size) {
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < size; ++row) {
        int r_up   = (row > 0) ? (row - 1) * size : -1;
        int r_curr = row * size;
        int r_down = (row < size - 1) ? (row + 1) * size : -1;

        for (int col = 0; col < size; ++col) {
            int c_left  = col - 1;
            int c_right = col + 1;
            int neighbours = 0;

            // Row above
            if (r_up != -1) {
                if (c_left >= 0) neighbours += current[r_up + c_left];
                neighbours += current[r_up + col];
                if (c_right < size) neighbours += current[r_up + c_right];
            }

            // Current row (left and right)
            if (c_left >= 0) neighbours += current[r_curr + c_left];
            if (c_right < size) neighbours += current[r_curr + c_right];

            // Row below
            if (r_down != -1) {
                if (c_left >= 0) neighbours += current[r_down + c_left];
                neighbours += current[r_down + col];
                if (c_right < size) neighbours += current[r_down + c_right];
            }

            int idx = r_curr + col;
            if (current[idx] == 1) {
                next[idx] = (neighbours == 2 || neighbours == 3) ? 1 : 0;
            } else {
                next[idx] = (neighbours == 3) ? 1 : 0;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    int size = DEFAULT_SIZE;
    int iterations = DEFAULT_ITERATIONS;

    if (argc > 1) {
        size = atoi(argv[1]);
        if (size <= 0) size = DEFAULT_SIZE;
    }
    if (argc > 2) {
        iterations = atoi(argv[2]);
        if (iterations <= 0) iterations = DEFAULT_ITERATIONS;
    }

    const size_t totalCells = static_cast<size_t>(size) * size;
    vector<int> current(totalCells, 0);
    vector<int> next(totalCells, 0);

    // Deterministic seed matching CUDA implementation for cross-verification
    mt19937 generator(42);
    uniform_int_distribution<int> distribution(0, 1);

    for (size_t i = 0; i < totalCells; ++i) {
        current[i] = distribution(generator);
    }

    #ifdef _OPENMP
    cout << "===== CPU GAME OF LIFE (OpenMP Multi-Core) =====" << endl;
    cout << "Threads: " << omp_get_max_threads() << endl;
    #else
    cout << "===== CPU GAME OF LIFE (Single-Core) =====" << endl;
    #endif
    cout << "Grid Size: " << size << " x " << size << endl;
    cout << "Iterations: " << iterations << endl;

    auto start = chrono::high_resolution_clock::now();

    for (int generation = 0; generation < iterations; ++generation) {
        nextGenerationOptimized(current.data(), next.data(), size);
        current.swap(next);
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    size_t liveCells = 0;
    for (size_t i = 0; i < totalCells; ++i) {
        liveCells += current[i];
    }

    double gigaUpdates = (static_cast<double>(totalCells) * iterations) / (elapsed.count() * 1e6);

    cout << "CPU Execution Time: " << elapsed.count() << " seconds" << endl;
    cout << "Throughput: " << gigaUpdates << " MegaUpdates/sec" << endl;
    cout << "Generations/sec: " << (iterations / elapsed.count()) << endl;
    cout << "Final Live Cells: " << liveCells << endl;

    return 0;
}