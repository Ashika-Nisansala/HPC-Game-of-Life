#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

constexpr int SIZE = 2000;
constexpr int ITERATIONS = 100;

void nextGeneration(const vector<vector<int>>& current, vector<vector<int>>& next) {
    for (int row = 0; row < SIZE; ++row) {
        for (int col = 0; col < SIZE; ++col) {
            int neighbours = 0;

            for (int rowOffset = -1; rowOffset <= 1; ++rowOffset) {
                for (int colOffset = -1; colOffset <= 1; ++colOffset) {
                    if (rowOffset == 0 && colOffset == 0) {
                        continue;
                    }

                    int neighbourRow = row + rowOffset;
                    int neighbourCol = col + colOffset;

                    if (neighbourRow >= 0 && neighbourRow < SIZE &&
                        neighbourCol >= 0 && neighbourCol < SIZE) {
                        neighbours += current[neighbourRow][neighbourCol];
                    }
                }
            }

            if (current[row][col] == 1) {
                next[row][col] = (neighbours == 2 || neighbours == 3) ? 1 : 0;
            } else {
                next[row][col] = (neighbours == 3) ? 1 : 0;
            }
        }
    }
}

int main() {
    vector<vector<int>> current(SIZE, vector<int>(SIZE, 0));
    vector<vector<int>> next(SIZE, vector<int>(SIZE, 0));

    random_device rd;
    mt19937 generator(rd());
    uniform_int_distribution<int> distribution(0, 1);

    for (int row = 0; row < SIZE; ++row) {
        for (int col = 0; col < SIZE; ++col) {
            current[row][col] = distribution(generator);
        }
    }

    auto start = chrono::high_resolution_clock::now();

    for (int generation = 0; generation < ITERATIONS; ++generation) {
        nextGeneration(current, next);
        current.swap(next);
        next.assign(SIZE, vector<int>(SIZE, 0));
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "Grid Size: " << SIZE << " x " << SIZE << endl;
    cout << "Iterations: " << ITERATIONS << endl;
    cout << "CPU Execution Time: " << elapsed.count() << " seconds" << endl;

    return 0;
}