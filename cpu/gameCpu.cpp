#include <iostream>
using namespace std;

int main()
{
    const int ROWS = 5;
    const int COLS = 5;

    int grid[ROWS][COLS] =
    {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0}
    };

    for (int row = 0; row < ROWS; row++)
    {
        for (int col = 0; col < COLS; col++)
        {
            cout << grid[row][col] << " ";
        }

        cout << endl;
    }

    return 0;
}