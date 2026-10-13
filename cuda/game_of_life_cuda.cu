
#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

constexpr int SIZE = 2000;
constexpr int ITERATIONS = 100;

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
__global__ void gameOfLifeKernel(const int *current, int *next)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= SIZE || col >= SIZE)
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
            if (neighbourRow >= 0 && neighbourRow < SIZE &&
                neighbourCol >= 0 && neighbourCol < SIZE)
            {
                neighbours += current[neighbourRow * SIZE + neighbourCol];
            }
        }
    }

    int index = row * SIZE + col;

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

int main()
{
    const size_t totalCells = static_cast<size_t>(SIZE) * SIZE;
    const size_t bytes = totalCells * sizeof(int);

    // Create initial grid on CPU
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

    // Configure CUDA threads and blocks
    dim3 blockSize(16, 16);

    dim3 gridSize(
        (SIZE + blockSize.x - 1) / blockSize.x,
        (SIZE + blockSize.y - 1) / blockSize.y);

    // CUDA timing events
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm up GPU before measurement
    gameOfLifeKernel<<<gridSize, blockSize>>>(d_current, d_next);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Start timing
    CUDA_CHECK(cudaEventRecord(start));

    for (int generation = 0; generation < ITERATIONS; generation++)
    {
        gameOfLifeKernel<<<gridSize, blockSize>>>(
            d_current,
            d_next);

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

    cout << "===== CUDA GAME OF LIFE =====" << endl;
    cout << "Grid Size: " << SIZE << " x " << SIZE << endl;
    cout << "Iterations: " << ITERATIONS << endl;
    cout << "Block Size: 16 x 16" << endl;
    cout << "GPU Execution Time: "
         << milliseconds / 1000.0f
         << " seconds" << endl;
    cout << "Final Live Cells: " << liveCells << endl;

    // Release GPU memory
    CUDA_CHECK(cudaFree(d_current));
    CUDA_CHECK(cudaFree(d_next));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
