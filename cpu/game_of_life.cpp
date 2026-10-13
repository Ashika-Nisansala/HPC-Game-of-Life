#include <iostream>
using namespace std;

const int ROWS = 2000;
const int COLS = 2000;

// Print the grid
void printGrid(int grid[ROWS][COLS])
{
    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            cout << grid[row][col] << " ";
        }

        cout << endl;
    }
}

// Count live neighbours
int countNeighbours(int grid[ROWS][COLS], int row, int col)
{
    int count = 0;

    for (int i = -1; i <= 1; i++)
    {
        for (int j = -1; j <= 1; j++)
        {
            // Don't count the cell itself
            if (i == 0 && j == 0)
                continue;

            int neighbourRow = row + i;
            int neighbourCol = col + j;

            // Check whether neighbour is inside the grid
            if (neighbourRow >= 0 && neighbourRow < ROWS &&
                neighbourCol >= 0 && neighbourCol < COLS)
            {
                count += grid[neighbourRow][neighbourCol];
            }
        }
    }

    return count;
}

// Calculate the next generation
void nextGeneration(int current[ROWS][COLS],
                    int next[ROWS][COLS])
{
    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            int neighbours = countNeighbours(current, row, col);

            // Game of Life rules
            if (current[row][col] == 1)
            {
                // Live cell survives with 2 or 3 neighbours
                if (neighbours == 2 || neighbours == 3)
                    next[row][col] = 1;
                else
                    next[row][col] = 0;
            }
            else
            {
                // Dead cell becomes alive with exactly 3 neighbours
                if (neighbours == 3)
                    next[row][col] = 1;
                else
                    next[row][col] = 0;
            }
        }
    }
}

int main()
{
    int current[ROWS][COLS] =
    {
        {0, 1, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {1, 1, 1, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    int next[ROWS][COLS];

    cout << "Initial Generation:" << endl;
    printGrid(current);

    // Run 4 generations
    for (int generation = 1; generation <= 4; generation++)
    {
        nextGeneration(current, next);

        // Copy next generation into current
        for (int row = 0; row < ROWS; row++)
        {
            for (int col = 0; col < COLS; col++)
            {
                current[row][col] = next[row][col];
            }
        }

        cout << "\nGeneration " << generation << ":" << endl;
        printGrid(current);
    }

    return 0;
}