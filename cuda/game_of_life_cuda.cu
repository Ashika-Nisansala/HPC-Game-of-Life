#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

constexpr int DEFAULT_SIZE = 2000;
constexpr int DEFAULT_ITERATIONS = 100;

// Check CUDA errors
#define CUDA_CHECK(call)                                        \
    do                                                          \
    {                                                           \
        cudaError_t error = (call);                             \
        if (error != cudaSuccess)                               \
        {                                                       \
            cerr << "CUDA Error: " << cudaGetErrorString(error) \
                 << " at line " << __LINE__ << endl;            \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while (0)

// Each GPU thread updates one cell
__global__ void gameOfLifeKernel(const int *current, int *next, int size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= size || col >= size)
        return;

    int neighbours = 0;

    // Count live neighbours
    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            if (i == 0 && j == 0)
                continue;

            int neighbourRow = row + i;
            int neighbourCol = col + j;

            // Boundary checking
            if (neighbourRow >= 0 && neighbourRow < size &&
                neighbourCol >= 0 && neighbourCol < size)
            {
                neighbours += current[neighbourRow * size + neighbourCol];
            }
        }
    }

    int index = row * size + col;

    // Apply Conway's Game of Life rules
    if (current[index] == 1)
    {
        next[index] = (neighbours == 2 || neighbours == 3) ? 1 : 0;
    }
    else
    {
        next[index] = (neighbours == 3) ? 1 : 0;
    }
}

int main(int argc, char *argv[])
{
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
    const size_t bytes = totalCells * sizeof(int);

    // Create initial grid on CPU with seed 42 matching CPU benchmark
    vector<int> h_grid(totalCells);

    mt19937 generator(42);
    uniform_int_distribution<int> distribution(0, 1);

    for (size_t i = 0; i < totalCells; i++)
    {
        h_grid[i] = distribution(generator);
    }

    // Allocate GPU memory
    int *d_current = nullptr;
    int *d_next = nullptr;

    CUDA_CHECK(cudaMalloc((void **)&d_current, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_next, bytes));

    // Transfer initial grid from CPU to GPU
    CUDA_CHECK(cudaMemcpy(
        d_current,
        h_grid.data(),
        bytes,
        cudaMemcpyHostToDevice));

    // Configure CUDA threads and blocks (16x16 2D thread block)
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (size + blockSize.x - 1) / blockSize.x,
        (size + blockSize.y - 1) / blockSize.y);

    // CUDA timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    for (int generation = 0; generation < iterations; generation++)
    {
        gameOfLifeKernel<<<gridSize, blockSize>>>(
            d_current,
            d_next,
            size);

        CUDA_CHECK(cudaGetLastError());

        // Swap GPU buffers
        int *temp = d_current;
        d_current = d_next;
        d_next = temp;
    }

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds,
        start,
        stop));

    // Copy final grid back to CPU
    CUDA_CHECK(cudaMemcpy(
        h_grid.data(),
        d_current,
        bytes,
        cudaMemcpyDeviceToHost));

    // Count remaining live cells
    size_t liveCells = 0;
    for (size_t i = 0; i < totalCells; i++)
    {
        liveCells += h_grid[i];
    }

    double elapsedSeconds = milliseconds / 1000.0;
    double megaUpdates = (static_cast<double>(totalCells) * iterations) / (elapsedSeconds * 1e6);

    cout << "===== CUDA GAME OF LIFE =====" << endl;
    cout << "Grid Size: " << size << " x " << size << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Block Size: 16 x 16" << endl;
    cout << "GPU Execution Time: " << elapsedSeconds << " seconds" << endl;
    cout << "Throughput: " << megaUpdates << " MegaUpdates/sec" << endl;
    cout << "Generations/sec: " << (iterations / elapsedSeconds) << endl;
    cout << "Final Live Cells: " << liveCells << endl;

    // Release GPU memory
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
