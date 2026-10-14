"""
HPC - Conway's Game of Life: Master Entry Point
================================================
Unified controller to launch GUI visualizer, compile and run optimized
CPU benchmarks (OpenMP multi-core), CUDA GPU benchmarks, or compare
performance across different computing architectures.

Usage:
    python main.py                  # Launch Interactive GUI Visualizer (Default)
    python main.py --gui            # Explicitly launch GUI
    python main.py --benchmark      # Run Python CPU simulation benchmark
    python main.py --cpu            # Compile & run C++ CPU benchmark (OpenMP)
    python main.py --demo           # Compile & run C++ Console Glider demo
    python main.py --cuda           # Compile & run CUDA GPU benchmark
    python main.py --compare        # Run multi-platform performance comparison
    python main.py --menu           # Interactive terminal selection menu
"""

import sys
import os
import argparse
import subprocess
import shutil
import time
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).parent.resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gui.game_of_life_gui import launch_gui, run_cpu_benchmark


def print_banner():
    banner = r"""
======================================================================
     _   _ ____   ____    ____                            __   _     _  __     
    | | | |  _ \ / ___|  / ___| __ _ _ __ ___   ___     ___ / _| |   (_)/ _|___ 
    | |_| | |_) | |     | |  _ / _` | '_ ` _ \ / _ \   / _ \ |_| |   | | |_/ _ \
    |  _  |  __/| |___  | |_| | (_| | | | | | |  __/  | (_) |  _| |___| |  _|  __/
    |_| |_|_|    \____|  \____|\__,_|_| |_| |_|\___|   \___/|_| |_____|_|_|  \___|
======================================================================
  High Performance Computing - Conway's Game of Life
  Platforms: Multi-Core CPU (OpenMP) | GPU (CUDA) | Visualizer (Tkinter)
======================================================================
"""
    print(banner)


def check_tool(name):
    """Check if a CLI executable is available in PATH."""
    return shutil.which(name) is not None


def ensure_build_dir():
    """Ensure the build/ output directory exists."""
    build_dir = REPO_ROOT / "build"
    build_dir.mkdir(exist_ok=True)
    return build_dir


def run_gui():
    """Launch the Tkinter GUI visualizer."""
    print("[*] Launching HPC Game of Life GUI Visualizer...")
    try:
        launch_gui()
    except Exception as e:
        print(f"[!] Failed to launch GUI: {e}")
        print("[*] Falling back to interactive terminal menu...")
        interactive_menu()


def run_py_benchmark(size=500, iters=100):
    """Execute pure Python optimized simulation benchmark."""
    print(f"\n[*] Running Optimized Python Simulation Benchmark ({size}x{size}, {iters} iterations)...")
    res = run_cpu_benchmark(rows=size, cols=size, iterations=iters)

    print("-" * 55)
    print(f"  Backend:             Python (Optimized Branchless)")
    print(f"  Grid Size:           {res['rows']} x {res['cols']} ({res['rows']*res['cols']:,} cells)")
    print(f"  Iterations:          {res['iterations']:,}")
    print(f"  Elapsed Time:        {res['elapsed_seconds']:.4f} seconds")
    print(f"  Generations/sec:     {res['generations_per_sec']:.2f}")
    print(f"  Throughput:          {res['mega_updates_per_sec']:.2f} MegaUpdates/sec")
    print(f"  Final Live Cells:    {res['live_cells']:,}")
    print("-" * 55)
    return res


def run_cpp_cpu_benchmark(size=2000, iters=100):
    """Compile and run the C++ OpenMP multi-core benchmark."""
    src = REPO_ROOT / "cpu" / "cpu_game_of_life.cpp"
    if not src.exists():
        print(f"[!] Error: Source file not found: {src}")
        return None

    if not check_tool("g++"):
        print("[!] Error: 'g++' compiler not found in PATH.")
        print("    Please install MinGW/GCC to compile and run C++ benchmarks.")
        return None

    build_dir = ensure_build_dir()
    exe = build_dir / ("cpu_benchmark.exe" if os.name == "nt" else "cpu_benchmark")

    # Compile with -O3 and OpenMP support
    print(f"[*] Compiling {src.name} with g++ (-O3 -fopenmp)...")
    compile_cmd = ["g++", "-O3", "-fopenmp", str(src), "-o", str(exe)]
    comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)

    if comp_res.returncode != 0:
        # Fallback without OpenMP if -fopenmp fails
        print("[!] OpenMP flag warning, trying standard compilation (-O3)...")
        compile_cmd = ["g++", "-O3", str(src), "-o", str(exe)]
        comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
        if comp_res.returncode != 0:
            print(f"[!] Compilation failed:\n{comp_res.stderr}")
            return None

    print(f"[*] Running C++ CPU Benchmark ({size}x{size}, {iters} iterations)...")
    start_t = time.perf_counter()
    run_res = subprocess.run([str(exe), str(size), str(iters)], capture_output=True, text=True)
    total_time = time.perf_counter() - start_t

    if run_res.returncode != 0:
        print(f"[!] Execution failed:\n{run_res.stderr}")
        return None

    print(run_res.stdout.strip())
    return {"output": run_res.stdout, "wall_time": total_time}


def run_cpp_demo():
    """Compile and run the 5x5 C++ console glider demo."""
    src = REPO_ROOT / "cpu" / "game_of_life.cpp"
    if not src.exists():
        print(f"[!] Error: Source file not found: {src}")
        return

    if not check_tool("g++"):
        print("[!] Error: 'g++' compiler not found in PATH.")
        return

    build_dir = ensure_build_dir()
    exe = build_dir / ("game_of_life_demo.exe" if os.name == "nt" else "game_of_life_demo")

    print(f"[*] Compiling {src.name} demo with g++...")
    comp_res = subprocess.run(["g++", "-O2", str(src), "-o", str(exe)], capture_output=True, text=True)
    if comp_res.returncode != 0:
        print(f"[!] Compilation failed:\n{comp_res.stderr}")
        return

    print("[*] Running C++ Console Glider Demo:\n")
    subprocess.run([str(exe)])


def run_cuda_benchmark(size=2000, iters=100):
    """Compile and run CUDA GPU benchmark."""
    src = REPO_ROOT / "cuda" / "game_of_life_cuda.cu"
    if not src.exists():
        print(f"[!] Error: Source file not found: {src}")
        return None

    if not check_tool("nvcc"):
        print("\n[!] NVIDIA CUDA Compiler ('nvcc') was not found in PATH.")
        print("    To run GPU acceleration:")
        print("    1. Ensure an NVIDIA GPU is present.")
        print("    2. Install NVIDIA CUDA Toolkit (https://developer.nvidia.com/cuda-downloads).")
        print("    3. Add nvcc to your system environment PATH.")
        return None

    build_dir = ensure_build_dir()
    exe = build_dir / ("cuda_benchmark.exe" if os.name == "nt" else "cuda_benchmark")

    print(f"[*] Compiling {src.name} with nvcc (-O3)...")
    comp_cmd = ["nvcc", "-O3", str(src), "-o", str(exe)]
    comp_res = subprocess.run(comp_cmd, capture_output=True, text=True)
    if comp_res.returncode != 0:
        print(f"[!] CUDA Compilation failed:\n{comp_res.stderr}")
        return None

    print(f"[*] Running CUDA Benchmark ({size}x{size}, {iters} iterations)...")
    run_res = subprocess.run([str(exe), str(size), str(iters)], capture_output=True, text=True)
    if run_res.returncode != 0:
        print(f"[!] CUDA execution failed:\n{run_res.stderr}")
        return None

    print(run_res.stdout.strip())
    return {"output": run_res.stdout}


def run_comparison(size=1000, iters=100):
    """Run performance comparison between Python, C++ OpenMP, and CUDA."""
    print_banner()
    print(f"[*] Starting Multi-Platform Performance Comparison")
    print(f"    Grid Size: {size} x {size} ({size*size:,} cells)")
    print(f"    Iterations: {iters}\n")

    results = []

    # 1. Python
    py_res = run_cpu_benchmark(rows=size, cols=size, iterations=iters)
    results.append({
        "platform": "Python (Branchless)",
        "time": py_res["elapsed_seconds"],
        "mega_updates": py_res["mega_updates_per_sec"],
        "gps": py_res["generations_per_sec"]
    })

    # 2. C++ OpenMP
    print("\n" + "=" * 55)
    cpp_res = run_cpp_cpu_benchmark(size=size, iters=iters)
    if cpp_res:
        # Parse elapsed time from output
        t = None
        for line in cpp_res["output"].splitlines():
            if "CPU Execution Time:" in line:
                try:
                    t = float(line.split(":")[1].replace("seconds", "").strip())
                except ValueError:
                    pass
        if t is None:
            t = cpp_res["wall_time"]

        total_cells = size * size
        mega_up = (total_cells * iters) / (t * 1e6)
        results.append({
            "platform": "C++ (OpenMP Multi-Core)",
            "time": t,
            "mega_updates": mega_up,
            "gps": iters / t
        })

    # 3. CUDA GPU (if available)
    if check_tool("nvcc"):
        print("\n" + "=" * 55)
        cu_res = run_cuda_benchmark(size=size, iters=iters)
        if cu_res:
            t = None
            for line in cu_res["output"].splitlines():
                if "GPU Execution Time:" in line:
                    try:
                        t = float(line.split(":")[1].replace("seconds", "").strip())
                    except ValueError:
                        pass
            if t is not None:
                total_cells = size * size
                results.append({
                    "platform": "NVIDIA CUDA (GPU)",
                    "time": t,
                    "mega_updates": (total_cells * iters) / (t * 1e6),
                    "gps": iters / t
                })

    # Print Summary Table
    print("\n" + "=" * 70)
    print(f"{'Platform':<28} | {'Time (s)':<10} | {'MUpdates/s':<12} | {'Speedup':<8}")
    print("-" * 70)

    baseline_time = results[0]["time"] if results else 1.0
    for r in results:
        speedup = f"{baseline_time / r['time']:.2f}x"
        print(f"{r['platform']:<28} | {r['time']:<10.4f} | {r['mega_updates']:<12.2f} | {speedup:<8}")
    print("=" * 70 + "\n")


def interactive_menu():
    """Display an interactive console selection menu."""
    while True:
        print_banner()
        print("Select an option:")
        print("  [1] Launch Interactive GUI Visualizer (Tkinter)")
        print("  [2] Run Python Simulation Benchmark")
        print("  [3] Compile & Run C++ CPU Benchmark (OpenMP Multi-Core)")
        print("  [4] Run C++ Console Glider Demo (Step-by-step 5x5)")
        print("  [5] Compile & Run CUDA GPU Benchmark (NVIDIA nvcc)")
        print("  [6] Run Multi-Platform Performance Comparison")
        print("  [0] Exit")
        print("-" * 55)

        try:
            choice = input("Enter choice [0-6] (default 1): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

        if choice in ("", "1"):
            run_gui()
            break
        elif choice == "2":
            run_py_benchmark(size=500, iters=100)
            input("\nPress Enter to return to menu...")
        elif choice == "3":
            run_cpp_cpu_benchmark(size=2000, iters=100)
            input("\nPress Enter to return to menu...")
        elif choice == "4":
            run_cpp_demo()
            input("\nPress Enter to return to menu...")
        elif choice == "5":
            run_cuda_benchmark(size=2000, iters=100)
            input("\nPress Enter to return to menu...")
        elif choice == "6":
            run_comparison(size=1000, iters=100)
            input("\nPress Enter to return to menu...")
        elif choice == "0":
            print("Exiting.")
            sys.exit(0)
        else:
            print("[!] Invalid option. Please enter a number between 0 and 6.")
            time.sleep(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HPC Conway's Game of Life: Master Controller & Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py                     # Launch GUI visualizer
  python main.py --benchmark         # Run Python CPU benchmark
  python main.py --cpu               # Run C++ OpenMP CPU benchmark (2000x2000)
  python main.py --cpu -s 1000 -i 50 # Run C++ CPU benchmark with custom size
  python main.py --cuda              # Run CUDA GPU benchmark
  python main.py --compare           # Compare CPU, Python, and CUDA speed
  python main.py --menu              # Interactive selection menu
"""
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-g", "--gui", action="store_true", help="Launch interactive GUI visualizer (default)")
    group.add_argument("-b", "--benchmark", action="store_true", help="Run Python CPU simulation benchmark")
    group.add_argument("-c", "--cpu", action="store_true", help="Compile and run C++ OpenMP multi-core benchmark")
    group.add_argument("-d", "--demo", action="store_true", help="Run C++ console glider step-by-step demo")
    group.add_argument("-k", "--cuda", action="store_true", help="Compile and run CUDA GPU benchmark")
    group.add_argument("--compare", action="store_true", help="Run multi-platform benchmark comparison")
    group.add_argument("-m", "--menu", action="store_true", help="Interactive terminal menu")

    parser.add_argument("-s", "--size", type=int, default=None, help="Grid size (width and height, e.g., 1000 or 2000)")
    parser.add_argument("-i", "--iters", type=int, default=None, help="Number of generations to simulate (e.g., 100)")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Determine size and iters defaults if specified
    size = args.size
    iters = args.iters

    if args.benchmark:
        run_py_benchmark(size=size or 500, iters=iters or 100)
    elif args.cpu:
        run_cpp_cpu_benchmark(size=size or 2000, iters=iters or 100)
    elif args.demo:
        run_cpp_demo()
    elif args.cuda:
        run_cuda_benchmark(size=size or 2000, iters=iters or 100)
    elif args.compare:
        run_comparison(size=size or 1000, iters=iters or 100)
    elif args.menu:
        interactive_menu()
    else:
        # Default behavior: Launch GUI
        # If no GUI display or user requested, launch GUI visualizer
        run_gui()


if __name__ == "__main__":
    main()
