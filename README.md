# HPC - Conway's Game of Life

A High Performance Computing (HPC) simulation and visualization suite for **Conway's Game of Life**, implementing:
- **Interactive GUI Visualizer** (Tkinter) with high-efficiency canvas updates, interactive cell painting, and preset patterns.
- **Optimized Multi-Core CPU Engine** (C++ with OpenMP & contiguous 1D memory layout).
- **GPU Acceleration** (NVIDIA CUDA with 2D thread block grid mapping).
- **Unified Master Controller** (`main.py`) to launch, benchmark, and compare all implementations.

---

## Quick Start

Launch the interactive visual dashboard directly:
```bash
python main.py
```

Or run an interactive terminal menu:
```bash
python main.py --menu
```

---

## Command-Line Usage

The project includes a unified CLI entry point via `main.py`:

| Command | Description |
|---|---|
| `python main.py` | Launch interactive Tkinter GUI Visualizer (Default) |
| `python main.py --gui` (`-g`) | Explicitly launch the GUI |
| `python main.py --benchmark` (`-b`) | Run pure Python simulation benchmark |
| `python main.py --cpu` (`-c`) | Compile & run C++ multi-core OpenMP benchmark (2000x2000) |
| `python main.py --demo` (`-d`) | Compile & run 5x5 C++ console step-by-step glider demo |
| `python main.py --cuda` (`-k`) | Compile & run NVIDIA CUDA GPU benchmark |
| `python main.py --compare` | Run multi-platform side-by-side performance comparison |
| `python main.py --menu` (`-m`) | Launch interactive console menu |

### Custom Grid Sizes and Iterations
You can pass custom dimensions and iteration counts to benchmarks:
```bash
# Run C++ OpenMP benchmark on a 1000x1000 grid for 50 iterations:
python main.py --cpu -s 1000 -i 50

# Run Python benchmark on a 300x300 grid for 50 iterations:
python main.py --benchmark -s 300 -i 50

# Compare platforms on a 500x500 grid for 100 iterations:
python main.py --compare -s 500 -i 100
```

---

## Architectural & Algorithmic Optimizations

### 1. Unified Master Controller (`main.py`)
- Replaces scattered, manual compiler calls with a central orchestrator.
- Automatically detects compilers (`g++`, `nvcc`), parallel threads, and build directories.
- Seamlessly builds and manages binaries in `build/`.

### 2. High-Performance GUI Visualizer (`gui/game_of_life_gui.py`)
- **Incremental Canvas Item Updates**: Replaces full `canvas.delete("all")` (which recreated 4,800 canvas objects every 60ms) with persistent canvas rectangle IDs. Only cells that flip state are updated (`canvas.itemconfig`), slashing Tkinter CPU overhead by ~90%.
- **Branchless Padded Boundary**: Zero-padded grid border allows 8-neighbor summation without inner coordinate checks, yielding a **~6x speedup** in Python.
- **Interactive Canvas Drawing**: Left-click or drag to paint alive cells, right-click or drag to erase cells.
- **Presets**: Gosper Glider Gun, Glider, Pulsar Oscillator, Pentadecathlon, Acorn Methuselah, and Random.
- **Real-time Metrics**: Live generations/sec (GPS / FPS) and live cell counters.

### 3. Optimized C++ CPU Engine (`cpu/cpu_game_of_life.cpp`)
- **Contiguous 1D Memory Layout**: Flat array (`row * SIZE + col`) eliminates pointer indirection and cache misses from `vector<vector<int>>`.
- **OpenMP Multi-Core Parallelization**: Multi-threaded execution across all CPU cores via `#pragma omp parallel for schedule(static)`.
- **Cross-Platform Compatibility**: Uses `#ifdef _OPENMP` for automatic fallback on systems without OpenMP.
- **Cross-Validation**: Uses deterministic pseudo-random seed matching CUDA for verification of final live cell counts.

### 4. GPU Acceleration (`cuda/game_of_life_cuda.cu`)
- Massively parallel 2D grid of 16x16 thread blocks.
- Buffer ping-ponging on GPU VRAM with zero host-device data transfers during iteration loops.
- Support for dynamic grid dimensions and iteration arguments.

---

## Requirements

- **Python 3.8+** with standard library (`tkinter` included).
- **C++ Compiler** (`g++` / MinGW with OpenMP) for multi-core CPU benchmarks.
- *(Optional)* **NVIDIA CUDA Toolkit** (`nvcc`) & NVIDIA GPU for CUDA acceleration.
