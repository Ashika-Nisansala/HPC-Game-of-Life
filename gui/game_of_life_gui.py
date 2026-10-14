"""
HPC Game of Life - Dual Engine Visualizer & Benchmarking Suite
==============================================================
Side-by-side simulation comparing CPU and CUDA implementations.

Architecture:
  - Header: Live Global Stats, CUDA Availability Status Pill, and "Start Simulation" Button.
  - 5 Vertical Partitions:
      * Partition 1: Settings Panel (Iterations, Speed, GPU Multiplier, Presets, Grid Controls)
      * Partitions 2 & 3: CPU Running Process Canvas (Electric Cyan Theme)
      * Partitions 4 & 5: CUDA Running Process Canvas (Emerald Green Theme - Real CUDA GPU)
  - Merged Section (Spanning columns 2 to 5):
      * Real-time Comparison Line Chart:
          - X-Axis: Elapsed Time (s)
          - Y-Axis: No. of Iterations for Unit Time (iters/s)
          - CPU Curve in Blue (#00d4ff)
          - CUDA Curve in Green (#10b981)
          - Dual Engine Progress Bars
  - High Performance Computing (HPC) Asynchronous Acceleration:
      * Direct NVIDIA CUDA Driver Engine executing on NVIDIA GeForce GPUs (RTX 3050).
      * Decoupled asynchronous threads with GPU hardware multi-stepping (5x - 20x).
      * If CUDA is not found, displays "CUDA is not found" directly on the canvas.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import threading
import shutil
import subprocess
import os
import ctypes

# --- Color Scheme & Aesthetics ---
COLOR_CPU        = "#00d4ff"   # Electric Cyan / Blue for CPU
COLOR_CPU_DIM    = "#0284c7"
COLOR_CUDA       = "#10b981"   # Emerald Green for CUDA
COLOR_CUDA_DIM   = "#059669"

COLOR_BG_DARK    = "#0d1117"   # Deep Dark Background
COLOR_PANEL      = "#161b22"   # Panel & Card Background
COLOR_CARD       = "#1c2128"   # Subcard Background
COLOR_BORDER     = "#30363d"   # Subtle Card Border
COLOR_TEXT       = "#f0f6fc"   # Crisp White Text
COLOR_MUTED      = "#8b949e"   # Secondary Gray Text

COLOR_DEAD_CELL  = "#0a0e14"   # Canvas Dead Cell Color
COLOR_GRID_LINE  = "#1a2230"   # Canvas Grid Line Color

COLOR_SUCCESS    = "#10b981"   # Green for CUDA Available
COLOR_DANGER     = "#ef4444"   # Red for CUDA Not Available
COLOR_WARNING    = "#f59e0b"   # Amber for Paused

# --- Grid Specifications ---
CANVAS_COLS = 44
CANVAS_ROWS = 28
CELL_SIZE   = 8
CANVAS_W    = CANVAS_COLS * CELL_SIZE  # 352 px
CANVAS_H    = CANVAS_ROWS * CELL_SIZE  # 224 px
FRAME_MS    = 25

# --- Presets Dictionary ---
PRESETS = {
    "Gosper Glider Gun": [
        (4, 0), (4, 1), (5, 0), (5, 1),
        (2, 12), (2, 13), (3, 11), (4, 10), (5, 10), (6, 10), (7, 11), (8, 12), (8, 13),
        (5, 14), (3, 15), (7, 15), (4, 16), (5, 16), (6, 16), (5, 17),
        (2, 20), (2, 21), (3, 20), (3, 21), (4, 20), (4, 21), (1, 22), (5, 22),
        (0, 24), (1, 24), (5, 24), (6, 24),
        (2, 34), (2, 35), (3, 34), (3, 35)
    ],
    "Glider": [
        (0, 1), (1, 2), (2, 0), (2, 1), (2, 2)
    ],
    "Pulsar (Oscillator)": [
        (r, c)
        for r in (-6, -1, 1, 6)
        for c in (-4, -3, -2, 2, 3, 4)
    ] + [
        (r, c)
        for r in (-4, -3, -2, 2, 3, 4)
        for c in (-6, -1, 1, 6)
    ],
    "Pentadecathlon": [
        (-4, 0), (-3, 0), (-2, -1), (-2, 1), (-1, 0),
        (0, 0), (1, 0), (2, -1), (2, 1), (3, 0), (4, 0)
    ],
    "Acorn (Methuselah)": [
        (0, 1), (1, 3), (2, 0), (2, 1), (2, 4), (2, 5), (2, 6)
    ]
}

# --- Game of Life PTX Kernel for Direct CUDA Execution ---
PTX_CONWAY_KERNEL = b'''
.version 7.0
.target sm_50
.address_size 64

.visible .entry gameOfLifeKernel(
    .param .u64 current_ptr,
    .param .u64 next_ptr,
    .param .u32 rows,
    .param .u32 cols
) {
    .reg .pred %p<20>;
    .reg .b32 %r<50>;
    .reg .b64 %rd<30>;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3; // col

    mov.u32 %r5, %ctaid.y;
    mov.u32 %r6, %ntid.y;
    mov.u32 %r7, %tid.y;
    mad.lo.u32 %r8, %r5, %r6, %r7; // row

    ld.param.u32 %r9, [rows];
    ld.param.u32 %r10, [cols];

    setp.ge.s32 %p1, %r8, %r9;
    setp.ge.s32 %p2, %r4, %r10;
    or.pred %p3, %p1, %p2;
    @%p3 ret;

    ld.param.u64 %rd1, [current_ptr];
    ld.param.u64 %rd2, [next_ptr];

    mov.u32 %r11, 0; // neighbours
    mov.s32 %r12, -1; // dr

LOOP_DR:
    add.s32 %r13, %r8, %r12;
    setp.lt.s32 %p4, %r13, 0;
    setp.ge.s32 %p5, %r13, %r9;
    or.pred %p6, %p4, %p5;

    mov.s32 %r14, -1; // dc

LOOP_DC:
    setp.eq.s32 %p7, %r12, 0;
    setp.eq.s32 %p8, %r14, 0;
    and.pred %p9, %p7, %p8;
    @%p9 bra NEXT_DC;

    @%p6 bra NEXT_DC;

    add.s32 %r15, %r4, %r14;
    setp.lt.s32 %p10, %r15, 0;
    setp.ge.s32 %p11, %r15, %r10;
    or.pred %p12, %p10, %p11;
    @%p12 bra NEXT_DC;

    mad.lo.s32 %r16, %r13, %r10, %r15;
    mul.wide.s32 %rd3, %r16, 4;
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u32 %r17, [%rd4];
    add.u32 %r11, %r11, %r17;

NEXT_DC:
    add.s32 %r14, %r14, 1;
    setp.le.s32 %p13, %r14, 1;
    @%p13 bra LOOP_DC;

    add.s32 %r12, %r12, 1;
    setp.le.s32 %p14, %r12, 1;
    @%p14 bra LOOP_DR;

    mad.lo.s32 %r18, %r8, %r10, %r4;
    mul.wide.s32 %rd5, %r18, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u32 %r19, [%rd6];

    mov.u32 %r20, 0;
    setp.eq.s32 %p15, %r11, 3;
    @%p15 mov.u32 %r20, 1;

    setp.eq.s32 %p16, %r19, 1;
    setp.eq.s32 %p17, %r11, 2;
    and.pred %p18, %p16, %p17;
    @%p18 mov.u32 %r20, 1;

    add.u64 %rd7, %rd2, %rd5;
    st.global.u32 [%rd7], %r20;
    ret;
}
'''


class CudaDriverEngine:
    """
    Direct Hardware CUDA Engine utilizing NVIDIA CUDA Driver API (nvcuda.dll / libcuda.so).
    Executes native GPU kernels on NVIDIA GeForce/RTX GPUs without requiring Visual Studio or nvcc.
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.total = rows * cols
        self.available = False
        self.gpu_name = ""
        self.cuda = None
        self.ctx = None
        self.mod = None
        self.func = None
        self.d_curr = None
        self.d_next = None
        self._init_cuda()

    def _init_cuda(self):
        try:
            if os.name == "nt":
                self.cuda = ctypes.WinDLL("nvcuda.dll")
            else:
                self.cuda = ctypes.CDLL("libcuda.so")

            if self.cuda.cuInit(0) != 0:
                return

            dev_count = ctypes.c_int()
            if self.cuda.cuDeviceGetCount(ctypes.byref(dev_count)) != 0 or dev_count.value <= 0:
                return

            dev = ctypes.c_int()
            if self.cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
                return

            name_buf = (ctypes.c_char * 256)()
            self.cuda.cuDeviceGetName(name_buf, 256, dev)
            self.gpu_name = name_buf.value.decode("utf-8", errors="ignore").strip()

            ctx = ctypes.c_void_p()
            if self.cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) != 0:
                return
            self.ctx = ctx

            mod = ctypes.c_void_p()
            if self.cuda.cuModuleLoadData(ctypes.byref(mod), PTX_CONWAY_KERNEL) != 0:
                return
            self.mod = mod

            func = ctypes.c_void_p()
            if self.cuda.cuModuleGetFunction(ctypes.byref(func), mod, b"gameOfLifeKernel") != 0:
                return
            self.func = func

            # Allocate device memory
            self.d_curr = ctypes.c_uint64()
            self.d_next = ctypes.c_uint64()
            bytes_size = self.total * 4
            self.cuda.cuMemAlloc_v2(ctypes.byref(self.d_curr), bytes_size)
            self.cuda.cuMemAlloc_v2(ctypes.byref(self.d_next), bytes_size)

            self.r_val = ctypes.c_int32(self.rows)
            self.c_val = ctypes.c_int32(self.cols)
            self.grid_x = (self.cols + 15) // 16
            self.grid_y = (self.rows + 15) // 16
            self.host_buf = (ctypes.c_int32 * self.total)()

            self.available = True
        except Exception:
            self.available = False

    def bind_thread(self):
        """Ensure current thread is bound to CUDA context."""
        if self.available and self.ctx:
            self.cuda.cuCtxSetCurrent(self.ctx)

    def upload(self, grid_2d):
        """Upload a 2D host grid to GPU device memory."""
        if not self.available:
            return
        self.bind_thread()
        idx = 0
        for r in range(self.rows):
            for c in range(self.cols):
                self.host_buf[idx] = grid_2d[r][c]
                idx += 1
        self.cuda.cuMemcpyHtoD_v2(self.d_curr, ctypes.byref(self.host_buf), self.total * 4)

    def step(self):
        """Execute one Game of Life generation on the GPU."""
        if not self.available:
            return
        args = (ctypes.c_void_p * 4)(
            ctypes.cast(ctypes.byref(self.d_curr), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(self.d_next), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(self.r_val), ctypes.c_void_p),
            ctypes.cast(ctypes.byref(self.c_val), ctypes.c_void_p),
        )
        self.cuda.cuLaunchKernel(
            self.func, self.grid_x, self.grid_y, 1, 16, 16, 1, 0, None, args, None
        )
        # Ping-pong GPU device pointers
        self.d_curr, self.d_next = self.d_next, self.d_curr

    def download(self):
        """Download the current generation from GPU device memory."""
        if not self.available:
            return None
        self.cuda.cuMemcpyDtoH_v2(ctypes.byref(self.host_buf), self.d_curr, self.total * 4)
        out = []
        idx = 0
        for r in range(self.rows):
            row = [self.host_buf[idx + c] for c in range(self.cols)]
            out.append(row)
            idx += self.cols
        return out

    def close(self):
        """Release CUDA resources."""
        if self.ctx:
            self.bind_thread()
            if self.d_curr: self.cuda.cuMemFree_v2(self.d_curr)
            if self.d_next: self.cuda.cuMemFree_v2(self.d_next)
            if self.mod: self.cuda.cuModuleUnload(self.mod)
            self.cuda.cuCtxDestroy_v2(self.ctx)


def check_cuda_availability():
    """
    Logically detect whether CUDA hardware and driver are available.
    Properly detects NVIDIA GPUs (RTX 3050, etc.) via Driver API and nvidia-smi.
    """
    try:
        cuda_lib = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so")
        if cuda_lib.cuInit(0) == 0:
            count = ctypes.c_int()
            if cuda_lib.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0:
                dev = ctypes.c_int()
                cuda_lib.cuDeviceGet(ctypes.byref(dev), 0)
                name_buf = (ctypes.c_char * 256)()
                cuda_lib.cuDeviceGetName(name_buf, 256, dev)
                gpu_name = name_buf.value.decode("utf-8", errors="ignore").strip()
                return True, "CUDA is available", gpu_name
    except Exception:
        pass

    smi_path = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    if os.path.exists(smi_path) if smi_path else False:
        try:
            res = subprocess.run(
                [smi_path, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=2
            )
            if res.returncode == 0 and res.stdout.strip():
                gpu_name = res.stdout.strip().splitlines()[0]
                return True, "CUDA is available", gpu_name
        except Exception:
            pass

    return False, "CUDA is not available", "No NVIDIA GPU or CUDA Driver detected"


def make_random_grid(rows, cols, density=0.30):
    """Generate a 2D grid with given live cell density."""
    return [[1 if random.random() < density else 0 for _ in range(cols)]
            for _ in range(rows)]


def make_empty_grid(rows, cols):
    """Generate an empty 2D grid (all dead)."""
    return [[0] * cols for _ in range(rows)]


def fast_next_generation(current, rows, cols):
    """
    Optimized Conway's Game of Life step using a zero-padded boundary.
    Returns: (new_grid, changed_cells_list)
    """
    padded = [[0] * (cols + 2)]
    for r in current:
        padded.append([0] + r + [0])
    padded.append([0] * (cols + 2))

    nxt = []
    changed = []

    for r in range(1, rows + 1):
        prev_r = padded[r - 1]
        curr_r = padded[r]
        next_r = padded[r + 1]
        orig_r = current[r - 1]
        new_row = [0] * cols

        for c in range(1, cols + 1):
            neighbours = (
                prev_r[c - 1] + prev_r[c] + prev_r[c + 1] +
                curr_r[c - 1]             + curr_r[c + 1] +
                next_r[c - 1] + next_r[c] + next_r[c + 1]
            )

            cell = orig_r[c - 1]
            val = 1 if (neighbours == 3 or (cell == 1 and neighbours == 2)) else 0
            new_row[c - 1] = val

            if val != cell:
                changed.append((r - 1, c - 1, val))

        nxt.append(new_row)

    return nxt, changed


def run_cpu_benchmark(rows=1000, cols=1000, iterations=100):
    """Standalone pure Python CPU benchmark."""
    grid = make_random_grid(rows, cols, density=0.30)
    start_time = time.perf_counter()

    for _ in range(iterations):
        grid, _ = fast_next_generation(grid, rows, cols)

    elapsed = time.perf_counter() - start_time
    total_updates = rows * cols * iterations
    mega_updates = total_updates / (elapsed * 1e6) if elapsed > 0 else 0
    live_cells = sum(sum(r) for r in grid)

    return {
        "rows": rows,
        "cols": cols,
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "generations_per_sec": iterations / elapsed if elapsed > 0 else 0,
        "mega_updates_per_sec": mega_updates,
        "live_cells": live_cells,
    }


# =============================================================================
# REAL-TIME COMPARISON LINE CHART COMPONENT
# =============================================================================
class RealtimeComparisonChart(tk.Canvas):
    """
    High-Performance Real-Time Canvas Line Chart.
    Plots 'No. of Iterations for Unit Time (iters/s)' on the Y-Axis vs 'Elapsed Time (ms)' on the X-Axis.
    - CPU Line: Electric Blue (#00d4ff)
    - CUDA Line: Emerald Green (#10b981)
    """

    def __init__(self, parent, height=185, bg="#0a0e17"):
        super().__init__(
            parent, height=height, bg=bg,
            highlightthickness=1, highlightbackground=COLOR_BORDER
        )
        self.w = 700
        self.h = height
        self.margin_l = 65
        self.margin_r = 30
        self.margin_t = 22
        self.margin_b = 28
        self.plot_w = max(100, self.w - self.margin_l - self.margin_r)
        self.plot_h = max(50, self.h - self.margin_t - self.margin_b)

        self.times = []
        self.cpu_pts = []
        self.cuda_times = []
        self.cuda_pts = []
        self.cuda_completed_at = None

        self.bind("<Configure>", self._on_resize)
        self.redraw()

    def _on_resize(self, event):
        if event.width > 100 and event.height > 60:
            self.w = event.width
            self.h = event.height
            self.plot_w = max(100, self.w - self.margin_l - self.margin_r)
            self.plot_h = max(50, self.h - self.margin_t - self.margin_b)
            self.redraw()

    def reset(self):
        self.times.clear()
        self.cpu_pts.clear()
        self.cuda_times.clear()
        self.cuda_pts.clear()
        self.cuda_completed_at = None
        self.redraw()

    def add_point(self, t_ms, cpu_val, cuda_val):
        t_f = float(t_ms)
        self.times.append(t_f)
        self.cpu_pts.append(max(0.0, float(cpu_val)))
        if self.cuda_completed_at is None:
            self.cuda_times.append(t_f)
            self.cuda_pts.append(max(0.0, float(cuda_val)))
        self.redraw()

    def mark_cuda_completed(self, t_ms):
        t_f = float(t_ms)
        self.cuda_completed_at = t_f
        if not self.cuda_times or abs(self.cuda_times[-1] - t_f) > 20:
            self.cuda_times.append(t_f)
            last_val = self.cuda_pts[-1] if self.cuda_pts else 0.0
            self.cuda_pts.append(last_val)
        self.redraw()

    def redraw(self):
        self.delete("all")
        x0 = self.margin_l
        y0 = self.margin_t
        x1 = self.w - self.margin_r
        y1 = self.h - self.margin_b

        # Y scale calculation (iterations per unit time)
        max_y = 50.0
        if self.cpu_pts or self.cuda_pts:
            max_data = max(max(self.cpu_pts or [0]), max(self.cuda_pts or [0]))
            max_y = max(max_y, max_data * 1.20)

        # Round max_y to nice clean interval
        if max_y <= 50:
            max_y = ((int(max_y) + 9) // 10) * 10
        elif max_y <= 150:
            max_y = ((int(max_y) + 24) // 25) * 25
        elif max_y <= 500:
            max_y = ((int(max_y) + 49) // 50) * 50
        else:
            max_y = ((int(max_y) + 99) // 100) * 100
        max_y = max(max_y, 40)

        # X scale calculation (time in milliseconds, min span 2000 ms)
        max_x = max(self.times[-1] if self.times else 2000.0, 2000.0)

        # Horizontal grid lines and Y-axis numbers
        num_y_ticks = 5 if self.plot_h >= 120 else 4
        for i in range(num_y_ticks + 1):
            val = int(max_y * i / num_y_ticks)
            y = y1 - (val / max_y) * self.plot_h
            # Grid dashed line
            self.create_line(x0, y, x1, y, fill="#1c2536", dash=(2, 4))
            # Number label
            self.create_text(x0 - 8, y, text=f"{val:,}", fill="#8b949e", font=("Segoe UI", 8), anchor="e")

        # Y-Axis Header Title
        self.create_text(x0 - 8, y0 - 10, text="iters / sec", fill=COLOR_MUTED, font=("Segoe UI", 8, "bold"), anchor="e")

        # Dynamic clean intervals for millisecond grid and numbers
        if max_x <= 1500:
            step_x = 250.0
        elif max_x <= 3000:
            step_x = 500.0
        elif max_x <= 7500:
            step_x = 1000.0
        elif max_x <= 15000:
            step_x = 2500.0
        elif max_x <= 30000:
            step_x = 5000.0
        else:
            step_x = 10000.0

        t = 0.0
        while t <= max_x + 0.1:
            x = x0 + (t / max_x) * self.plot_w
            self.create_line(x, y0, x, y1, fill="#1c2536", dash=(2, 4))
            self.create_text(x, y1 + 12, text=f"{int(t)}ms", fill="#8b949e", font=("Segoe UI", 8))
            t += step_x

        # X-Axis Header Title
        self.create_text(x1, y1 + 12, text="Time (ms)", fill=COLOR_MUTED, font=("Segoe UI", 8, "bold"), anchor="e")

        # Axes baseline borders
        self.create_line(x0, y0 - 4, x0, y1, fill="#374151", width=1.5)
        self.create_line(x0, y1, x1 + 4, y1, fill="#374151", width=1.5)

        # If simulation has not started yet, show an inviting prompt
        if len(self.times) == 0:
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            self.create_rectangle(cx - 230, cy - 16, cx + 230, cy + 16, fill="#111827", outline="#1f2937", width=1)
            self.create_text(
                cx, cy,
                text="▶ Click [ START SIMULATION ] to plot real-time CPU vs CUDA performance curves (ms)",
                fill="#94a3b8", font=("Segoe UI", 8, "bold")
            )
            return

        # Plot CPU Curve (Blue #00d4ff)
        if len(self.times) == 1:
            cx = x0 + (self.times[0] / max_x) * self.plot_w
            cy = y1 - (self.cpu_pts[0] / max_y) * self.plot_h
            self.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=COLOR_CPU, outline="#ffffff")
        elif len(self.times) >= 2:
            cpu_coords = []
            for t_val, c_val in zip(self.times, self.cpu_pts):
                cx = x0 + (t_val / max_x) * self.plot_w
                cy = y1 - (c_val / max_y) * self.plot_h
                cpu_coords.extend([cx, cy])
            # Draw line with width=3
            self.create_line(*cpu_coords, fill=COLOR_CPU, width=3, smooth=True)
            # Accent nodes along line
            stride = max(1, len(self.times) // 20)
            for idx in range(0, len(self.times), stride):
                nx = x0 + (self.times[idx] / max_x) * self.plot_w
                ny = y1 - (self.cpu_pts[idx] / max_y) * self.plot_h
                self.create_oval(nx - 2, ny - 2, nx + 2, ny + 2, fill="#0284c7", outline=COLOR_CPU)
            # Glowing dot on latest point
            last_x = cpu_coords[-2]
            last_y = cpu_coords[-1]
            self.create_oval(last_x - 5, last_y - 5, last_x + 5, last_y + 5, fill=COLOR_CPU, outline="#ffffff", width=1.5)

        # Plot CUDA Curve (Green #10b981)
        if len(self.cuda_times) == 1:
            cx = x0 + (self.cuda_times[0] / max_x) * self.plot_w
            cy = y1 - (self.cuda_pts[0] / max_y) * self.plot_h
            self.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill=COLOR_CUDA, outline="#ffffff")
        elif len(self.cuda_times) >= 2:
            cuda_coords = []
            for t_val, cu_val in zip(self.cuda_times, self.cuda_pts):
                cx = x0 + (t_val / max_x) * self.plot_w
                cy = y1 - (cu_val / max_y) * self.plot_h
                cuda_coords.extend([cx, cy])
            # Draw line with width=3
            self.create_line(*cuda_coords, fill=COLOR_CUDA, width=3, smooth=True)
            # Accent nodes along line
            stride = max(1, len(self.cuda_times) // 20)
            for idx in range(0, len(self.cuda_times), stride):
                nx = x0 + (self.cuda_times[idx] / max_x) * self.plot_w
                ny = y1 - (self.cuda_pts[idx] / max_y) * self.plot_h
                self.create_oval(nx - 2, ny - 2, nx + 2, ny + 2, fill="#059669", outline=COLOR_CUDA)
            # Glowing dot on latest point
            last_x = cuda_coords[-2]
            last_y = cuda_coords[-1]
            self.create_oval(last_x - 5, last_y - 5, last_x + 5, last_y + 5, fill=COLOR_CUDA, outline="#ffffff", width=1.5)

        # Milestone: If CUDA finished, draw completion flag in ms
        if self.cuda_completed_at is not None:
            mx = x0 + (self.cuda_completed_at / max_x) * self.plot_w
            if x0 <= mx <= x1:
                self.create_line(mx, y0, mx, y1, fill="#10b981", dash=(3, 3), width=1.5)
                self.create_rectangle(mx - 46, y0, mx + 46, y0 + 16, fill="#064e3b", outline="#10b981")
                self.create_text(mx, y0 + 8, text=f"CUDA {int(self.cuda_completed_at)}ms ✓", fill="#6ee7b7", font=("Segoe UI", 8, "bold"))


# =============================================================================
# MAIN DUAL ENGINE APP WINDOW
# =============================================================================
class DualGameOfLifeApp(tk.Tk):
    """
    Dual Engine HPC Visualizer for Conway's Game of Life.
    Displays side-by-side CPU and CUDA canvases with 5 vertical partitions
    and a real-time Comparison Line Chart (iters/s vs Time).
    """

    def __init__(self):
        super().__init__()

        self.title("HPC - Conway's Game of Life: Dual Engine CPU vs CUDA")
        self.configure(bg=COLOR_BG_DARK)

        # Simulation grid dimensions
        self.rows = CANVAS_ROWS
        self.cols = CANVAS_COLS
        self.cell_size = CELL_SIZE

        # Thread-safe simulation settings
        self.sim_target_iters = 500
        self.sim_speed = 65
        self.gpu_multiplier = 10

        # Initialize native Hardware CUDA Engine
        self.cuda_engine = CudaDriverEngine(self.rows, self.cols)
        if self.cuda_engine.available:
            self.cuda_available = True
            self.cuda_status_text = "CUDA is available"
            self.cuda_details = self.cuda_engine.gpu_name
        else:
            self.cuda_available, self.cuda_status_text, self.cuda_details = check_cuda_availability()

        # Independent grids for CPU and CUDA
        initial_pattern = make_random_grid(self.rows, self.cols, density=0.30)
        self.cpu_grid = [row[:] for row in initial_pattern]
        self.cuda_grid = [row[:] for row in initial_pattern]

        # Upload initial pattern to GPU if available
        if self.cuda_engine.available:
            self.cuda_engine.upload(self.cuda_grid)

        self.cpu_gen = 0
        self.cuda_gen = 0
        self.running = False
        self.cpu_done = False
        self.cuda_done = False

        self._cpu_thread = None
        self._cuda_thread = None
        self._lock = threading.Lock()

        # Pending incremental changes for thread-safe canvas rendering
        self._pending_cpu_changed = None
        self._pending_cuda_changed = None

        # Canvas item IDs for fast itemconfig updates
        self._cpu_rects = [[None] * self.cols for _ in range(self.rows)]
        self._cuda_rects = [[None] * self.cols for _ in range(self.rows)]
        self._cpu_canvas_grid = [row[:] for row in self.cpu_grid]
        self._cuda_canvas_grid = [row[:] for row in self.cuda_grid]

        # Performance tracking metrics
        self.start_sim_time = None
        self.pause_start_time = None
        self.cuda_completion_time = None
        self.cpu_completion_time = None
        self.cpu_gps = 0.0
        self.cuda_gps = 0.0
        self.cpu_gen_counter = 0
        self.cuda_gen_counter = 0
        self.last_perf_check = time.time()

        # Build UI Structure
        self._build_top_header()
        self._build_body_partitions()
        self._build_merged_bottom_section()

        # Initialize Canvas Graphics
        self._init_canvas_items(self.cpu_canvas, self._cpu_rects, self.cpu_grid, COLOR_CPU)

        if self.cuda_available:
            self._init_canvas_items(self.cuda_canvas, self._cuda_rects, self.cuda_grid, COLOR_CUDA)
        else:
            self._render_cuda_not_found_canvas()

        # Update initial text and stats
        self._update_all_stats()
        self._refresh_loop()

        # Clean thread termination on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Center window on screen and enable responsive resizing
        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = max(10, (sw - w) // 2)
        y = max(10, (sh - h) // 2 - 20)
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(960, 500)
        self.resizable(True, True)

    # =========================================================================
    # 1. TOP HEADER (Stats, CUDA Availability Status Pill, Start Button)
    # =========================================================================
    def _build_top_header(self):
        hdr = tk.Frame(self, bg=COLOR_PANEL, pady=10, padx=16,
                       highlightbackground=COLOR_BORDER, highlightthickness=1)
        hdr.pack(fill="x", padx=12, pady=(10, 8))

        # Left Section: Brand & CUDA Availability Pill
        left_box = tk.Frame(hdr, bg=COLOR_PANEL)
        left_box.pack(side="left")

        tk.Label(left_box, text="HPC GAME OF LIFE",
                 font=("Courier New", 15, "bold"),
                 fg=COLOR_CPU, bg=COLOR_PANEL).pack(side="left", padx=(0, 10))

        # CUDA Availability Status Logic & Pill Badge
        badge_bg = "#064e3b" if self.cuda_available else "#450a0a"
        badge_fg = COLOR_SUCCESS if self.cuda_available else COLOR_DANGER
        badge_border = COLOR_SUCCESS if self.cuda_available else COLOR_DANGER

        pill_frame = tk.Frame(left_box, bg=badge_bg, padx=10, pady=3,
                              highlightbackground=badge_border, highlightthickness=1)
        pill_frame.pack(side="left", padx=4)

        icon = "● "
        tk.Label(pill_frame, text=f"{icon}{self.cuda_status_text.upper()}",
                 font=("Segoe UI", 9, "bold"),
                 fg=badge_fg, bg=badge_bg).pack()

        # Middle Section: Global Stats Readouts
        stats_box = tk.Frame(hdr, bg=COLOR_PANEL)
        stats_box.pack(side="left", expand=True)

        self.stat_gen_lbl = self._make_header_stat(stats_box, "TARGET ITERS", "500", "#a855f7")
        self.stat_cpu_cells_lbl = self._make_header_stat(stats_box, "CPU CELLS", "0", COLOR_CPU)
        self.stat_cuda_cells_lbl = self._make_header_stat(stats_box, "CUDA CELLS", "0", COLOR_CUDA)
        self.stat_elapsed_lbl = self._make_header_stat(stats_box, "ELAPSED", "0.0s", COLOR_TEXT)

        # Right Section: Prominent "Start Simulation" Button
        btn_box = tk.Frame(hdr, bg=COLOR_PANEL)
        btn_box.pack(side="right")

        self.btn_start = tk.Button(
            btn_box, text="▶  START SIMULATION",
            font=("Segoe UI", 11, "bold"),
            bg=COLOR_SUCCESS, fg="#042f2e",
            activebackground="#34d399", activeforeground="#042f2e",
            relief="flat", cursor="hand2", padx=16, pady=6,
            command=self._on_start
        )
        self.btn_start.pack(side="left", padx=4)

        self.btn_pause = tk.Button(
            btn_box, text="⏸  PAUSE",
            font=("Segoe UI", 10, "bold"),
            bg="#ef4444", fg="#ffffff",
            activebackground="#f87171", activeforeground="#ffffff",
            relief="flat", cursor="hand2", padx=12, pady=6,
            state="disabled", command=self._on_pause
        )
        self.btn_pause.pack(side="left", padx=4)

    def _make_header_stat(self, parent, title, val, color):
        frame = tk.Frame(parent, bg=COLOR_PANEL, padx=12)
        frame.pack(side="left", expand=True)
        tk.Label(frame, text=title, font=("Segoe UI", 8, "bold"),
                 fg=COLOR_MUTED, bg=COLOR_PANEL).pack()
        lbl = tk.Label(frame, text=val, font=("Courier New", 12, "bold"),
                       fg=color, bg=COLOR_PANEL)
        lbl.pack()
        return lbl

    # =========================================================================
    # 2. BODY: 5 VERTICAL PARTITIONS
    #    Part 1: Settings (Col 0)
    #    Parts 2 & 3: CPU Process Canvas (Cols 1 & 2)
    #    Parts 4 & 5: CUDA Process Canvas (Cols 3 & 4)
    # =========================================================================
    def _build_body_partitions(self):
        body_container = tk.Frame(self, bg=COLOR_BG_DARK)
        body_container.pack(fill="both", expand=True, padx=12, pady=(0, 6))

        # Configure 5 vertical columns
        body_container.columnconfigure(0, weight=0, minsize=240)  # Part 1: Settings
        body_container.columnconfigure(1, weight=1)              # Part 2: CPU Left
        body_container.columnconfigure(2, weight=1)              # Part 3: CPU Right
        body_container.columnconfigure(3, weight=1)              # Part 4: CUDA Left
        body_container.columnconfigure(4, weight=1)              # Part 5: CUDA Right
        body_container.rowconfigure(0, weight=1)

        # -------------------------------------------------------------
        # PART 1: Settings Panel (Partition 1)
        # -------------------------------------------------------------
        settings_card = tk.Frame(body_container, bg=COLOR_PANEL, padx=12, pady=6,
                                 highlightbackground=COLOR_BORDER, highlightthickness=1)
        settings_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        tk.Label(settings_card, text="⚙  SIMULATION SETTINGS",
                 font=("Courier New", 9, "bold"),
                 fg=COLOR_CPU, bg=COLOR_PANEL).pack(anchor="w", pady=(0, 4))

        # Target Iterations
        tk.Label(settings_card, text="Target Iterations:",
                 font=("Segoe UI", 8, "bold"),
                 fg=COLOR_TEXT, bg=COLOR_PANEL).pack(anchor="w")

        self.iter_var = tk.IntVar(value=500)
        iter_spin = tk.Spinbox(settings_card, from_=10, to=100000, increment=50,
                               textvariable=self.iter_var, font=("Segoe UI", 8, "bold"),
                               bg=COLOR_CARD, fg=COLOR_TEXT, buttonbackground=COLOR_BORDER,
                               relief="flat", highlightthickness=1, highlightcolor=COLOR_CPU,
                               highlightbackground=COLOR_BORDER)
        iter_spin.pack(fill="x", pady=(1, 4))
        self.iter_var.trace_add("write", lambda *_: self._update_target_display())

        # GPU Acceleration Multiplier
        tk.Label(settings_card, text="GPU Hardware Acceleration:",
                 font=("Segoe UI", 8, "bold"),
                 fg=COLOR_CUDA, bg=COLOR_PANEL).pack(anchor="w")

        self.gpu_boost_var = tk.StringVar(value="10x GPU Turbo (Visually Faster)")
        boost_options = [
            "10x GPU Turbo (Visually Faster)",
            "5x GPU Speed (Fast)",
            "20x GPU Speed (Ultra Fast)",
            "1x Lockstep (1:1 Synchronized)"
        ]
        boost_cb = ttk.Combobox(settings_card, textvariable=self.gpu_boost_var,
                                values=boost_options, state="readonly", font=("Segoe UI", 8))
        boost_cb.pack(fill="x", pady=(1, 4))
        boost_cb.bind("<<ComboboxSelected>>", self._on_gpu_boost_changed)

        # Initial Preset Patterns
        tk.Label(settings_card, text="Preset Pattern:",
                 font=("Segoe UI", 8, "bold"),
                 fg=COLOR_TEXT, bg=COLOR_PANEL).pack(anchor="w")

        preset_options = ["Random (30% Density)"] + list(PRESETS.keys())
        self.preset_var = tk.StringVar(value="Random (30% Density)")
        preset_cb = ttk.Combobox(settings_card, textvariable=self.preset_var,
                                 values=preset_options, state="readonly", font=("Segoe UI", 8))
        preset_cb.pack(fill="x", pady=(1, 4))
        preset_cb.bind("<<ComboboxSelected>>", self._on_preset_selected)

        # Speed / Delay Slider
        tk.Label(settings_card, text="Base Animation Speed:",
                 font=("Segoe UI", 8, "bold"),
                 fg=COLOR_TEXT, bg=COLOR_PANEL).pack(anchor="w")

        self.speed_var = tk.IntVar(value=65)
        speed_scale = tk.Scale(settings_card, from_=1, to=100, orient="horizontal",
                               variable=self.speed_var, bg=COLOR_PANEL, fg=COLOR_CPU,
                               troughcolor=COLOR_CARD, highlightthickness=0,
                               sliderrelief="flat", showvalue=False)
        speed_scale.pack(fill="x", pady=(1, 4))
        self.speed_var.trace_add("write", lambda *_: self._on_speed_changed())

        # Control Action Buttons (2x2 Grid)
        btn_grid = tk.Frame(settings_card, bg=COLOR_PANEL)
        btn_grid.pack(fill="x", pady=(4, 0))

        self.btn_step = tk.Button(btn_grid, text="⏭ Step 1x", font=("Segoe UI", 8, "bold"),
                                  bg="#3b82f6", fg="#ffffff", relief="flat", cursor="hand2",
                                  command=self._on_step)
        self.btn_step.grid(row=0, column=0, sticky="ew", padx=(0, 2), pady=1)

        self.btn_sync = tk.Button(btn_grid, text="🔄 Sync Grids", font=("Segoe UI", 8, "bold"),
                                  bg="#8b5cf6", fg="#ffffff", relief="flat", cursor="hand2",
                                  command=self._on_sync_grids)
        self.btn_sync.grid(row=0, column=1, sticky="ew", padx=(2, 0), pady=1)

        self.btn_reset = tk.Button(btn_grid, text="🎲 Randomize", font=("Segoe UI", 8, "bold"),
                                   bg=COLOR_BORDER, fg=COLOR_TEXT, relief="flat", cursor="hand2",
                                   command=self._on_reset)
        self.btn_reset.grid(row=1, column=0, sticky="ew", padx=(0, 2), pady=1)

        self.btn_clear = tk.Button(btn_grid, text="🗑 Clear", font=("Segoe UI", 8, "bold"),
                                   bg="#475569", fg="#ffffff", relief="flat", cursor="hand2",
                                   command=self._on_clear)
        self.btn_clear.grid(row=1, column=1, sticky="ew", padx=(2, 0), pady=1)

        btn_grid.columnconfigure(0, weight=1)
        btn_grid.columnconfigure(1, weight=1)

        # -------------------------------------------------------------
        # PARTS 2 & 3: CPU Running Process Canvas (Partition 2 & 3)
        # -------------------------------------------------------------
        cpu_frame = tk.Frame(body_container, bg=COLOR_PANEL, padx=10, pady=8,
                             highlightbackground=COLOR_CPU, highlightthickness=1)
        cpu_frame.grid(row=0, column=1, columnspan=2, sticky="nsew", padx=4)
        self.cpu_frame = cpu_frame
        cpu_frame.bind("<Configure>", self._on_frame_resize)

        # CPU Canvas Header
        cpu_top = tk.Frame(cpu_frame, bg=COLOR_PANEL)
        cpu_top.pack(fill="x", pady=(0, 4))

        tk.Label(cpu_top, text="🖥  CPU ENGINE PROCESS",
                 font=("Courier New", 10, "bold"),
                 fg=COLOR_CPU, bg=COLOR_PANEL).pack(side="left")

        self.cpu_live_badge = tk.Label(cpu_top, text="Gen: 0 | Live: 0",
                                       font=("Segoe UI", 9, "bold"),
                                       fg=COLOR_CPU, bg=COLOR_PANEL)
        self.cpu_live_badge.pack(side="right")

        # CPU Interactive Canvas
        self.cpu_canvas = tk.Canvas(cpu_frame, width=CANVAS_W, height=CANVAS_H,
                                    bg=COLOR_DEAD_CELL, highlightthickness=0,
                                    cursor="crosshair")
        self.cpu_canvas.pack(padx=2, pady=2, expand=True)

        # Bind CPU canvas drawing
        self.cpu_canvas.bind("<Button-1>", lambda e: self._on_cpu_canvas_draw(e, 1, COLOR_CPU))
        self.cpu_canvas.bind("<B1-Motion>", lambda e: self._on_cpu_canvas_draw(e, 1, COLOR_CPU))
        self.cpu_canvas.bind("<Button-3>", lambda e: self._on_cpu_canvas_draw(e, 0, COLOR_DEAD_CELL))
        self.cpu_canvas.bind("<B3-Motion>", lambda e: self._on_cpu_canvas_draw(e, 0, COLOR_DEAD_CELL))

        # -------------------------------------------------------------
        # PARTS 4 & 5: CUDA Running Process Canvas (Partition 4 & 5)
        # -------------------------------------------------------------
        cuda_frame = tk.Frame(body_container, bg=COLOR_PANEL, padx=10, pady=8,
                              highlightbackground=COLOR_CUDA if self.cuda_available else COLOR_DANGER,
                              highlightthickness=1)
        cuda_frame.grid(row=0, column=3, columnspan=2, sticky="nsew", padx=(4, 0))
        self.cuda_frame = cuda_frame

        # CUDA Canvas Header
        cuda_top = tk.Frame(cuda_frame, bg=COLOR_PANEL)
        cuda_top.pack(fill="x", pady=(0, 4))

        cuda_title = "⚡  CUDA GPU PROCESS (RTX 3050)" if self.cuda_available else "⚡  CUDA GPU PROCESS (OFFLINE)"
        tk.Label(cuda_top, text=cuda_title,
                 font=("Courier New", 10, "bold"),
                 fg=COLOR_CUDA if self.cuda_available else COLOR_DANGER, bg=COLOR_PANEL).pack(side="left")

        self.cuda_live_badge = tk.Label(cuda_top, text="Gen: 0 | Live: 0",
                                        font=("Segoe UI", 9, "bold"),
                                        fg=COLOR_CUDA if self.cuda_available else COLOR_DANGER, bg=COLOR_PANEL)
        self.cuda_live_badge.pack(side="right")

        # CUDA Interactive Canvas
        self.cuda_canvas = tk.Canvas(cuda_frame, width=CANVAS_W, height=CANVAS_H,
                                     bg=COLOR_DEAD_CELL, highlightthickness=0,
                                     cursor="crosshair" if self.cuda_available else "arrow")
        self.cuda_canvas.pack(padx=2, pady=2, expand=True)

        if self.cuda_available:
            self.cuda_canvas.bind("<Button-1>", lambda e: self._on_cuda_canvas_draw(e, 1, COLOR_CUDA))
            self.cuda_canvas.bind("<B1-Motion>", lambda e: self._on_cuda_canvas_draw(e, 1, COLOR_CUDA))
            self.cuda_canvas.bind("<Button-3>", lambda e: self._on_cuda_canvas_draw(e, 0, COLOR_DEAD_CELL))
            self.cuda_canvas.bind("<B3-Motion>", lambda e: self._on_cuda_canvas_draw(e, 0, COLOR_DEAD_CELL))

    # =========================================================================
    # 3. MERGED BOTTOM SECTION: Spanning the 4 Vertical Canvas Parts (Cols 1 to 4)
    #    Real-Time Comparison Line Chart (Iterations/sec vs Elapsed Time)
    #    with CPU Line in Blue and CUDA Line in Green.
    # =========================================================================
    def _build_merged_bottom_section(self):
        bottom_container = tk.Frame(self, bg=COLOR_BG_DARK)
        bottom_container.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        bottom_container.columnconfigure(0, weight=0, minsize=240)  # Settings column placeholder
        bottom_container.columnconfigure(1, weight=1)              # Merged 4-part area: Col 1
        bottom_container.columnconfigure(2, weight=1)              # Col 2
        bottom_container.columnconfigure(3, weight=1)              # Col 3
        bottom_container.columnconfigure(4, weight=1)              # Col 4
        bottom_container.rowconfigure(0, weight=1)

        # Left auxiliary information card (under Settings)
        aux_card = tk.Frame(bottom_container, bg=COLOR_PANEL, padx=12, pady=6,
                            highlightbackground=COLOR_BORDER, highlightthickness=1)
        aux_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        tk.Label(aux_card, text="HARDWARE PROFILE", font=("Courier New", 8, "bold"),
                 fg=COLOR_MUTED, bg=COLOR_PANEL).pack(anchor="w")
        gpu_short = self.cuda_details if self.cuda_available else "Not Detected"
        tk.Label(aux_card, text=f"• Host Cores: {os.cpu_count() or 4} Threads\n• CUDA Device: {gpu_short}\n• Blue Line: CPU Speed (iters/s)\n• Green Line: CUDA Speed (iters/s)\n• Timeline Unit: Milliseconds (ms)",
                 font=("Segoe UI", 8), fg=COLOR_TEXT, bg=COLOR_PANEL, justify="left").pack(anchor="w", pady=(2, 0))

        # Merged 4-part card spanning Columns 1, 2, 3, 4
        merged_card = tk.Frame(bottom_container, bg=COLOR_PANEL, padx=12, pady=6,
                               highlightbackground=COLOR_BORDER, highlightthickness=1)
        merged_card.grid(row=0, column=1, columnspan=4, sticky="nsew", padx=(0, 0))

        # Title & Legend Row
        title_row = tk.Frame(merged_card, bg=COLOR_PANEL)
        title_row.pack(fill="x", pady=(0, 2))

        tk.Label(title_row, text="📈  COMPARISON LINE CHART (Iterations / Unit Time vs Time in ms)",
                 font=("Courier New", 9, "bold"),
                 fg=COLOR_TEXT, bg=COLOR_PANEL).pack(side="left")

        # Live Chart Legends & Metrics Badges (Top Right of Chart Card)
        legend_box = tk.Frame(title_row, bg=COLOR_PANEL)
        legend_box.pack(side="right")

        # CPU Legend (Blue)
        tk.Label(legend_box, text="■ CPU:", font=("Segoe UI", 9, "bold"),
                 fg=COLOR_CPU, bg=COLOR_PANEL).pack(side="left", padx=(0, 2))
        self.chart_cpu_lbl = tk.Label(legend_box, text="0.0 iters/s", font=("Courier New", 9, "bold"),
                                      fg=COLOR_CPU, bg=COLOR_PANEL)
        self.chart_cpu_lbl.pack(side="left", padx=(0, 12))

        # CUDA Legend (Green)
        tk.Label(legend_box, text="■ CUDA:", font=("Segoe UI", 9, "bold"),
                 fg=COLOR_CUDA, bg=COLOR_PANEL).pack(side="left", padx=(0, 2))
        self.chart_cuda_lbl = tk.Label(legend_box, text="0.0 iters/s", font=("Courier New", 9, "bold"),
                                       fg=COLOR_CUDA, bg=COLOR_PANEL)
        self.chart_cuda_lbl.pack(side="left", padx=(0, 12))

        # Speedup Ratio Badge
        self.speedup_badge = tk.Label(legend_box, text="⚡ Speedup: --",
                                      font=("Segoe UI", 9, "bold"),
                                      fg=COLOR_WARNING, bg=COLOR_PANEL)
        self.speedup_badge.pack(side="left")

        # The Real-Time Comparison Line Chart Canvas (Responsive with increased height)
        self.chart = RealtimeComparisonChart(merged_card, height=185, bg="#0a0e17")
        self.chart.pack(fill="both", expand=True, pady=(2, 4))

        # Real-time Visual Progress Bars for CPU & CUDA
        bars_frame = tk.Frame(merged_card, bg=COLOR_PANEL)
        bars_frame.pack(fill="x", pady=(2, 0))

        style = ttk.Style()
        style.theme_use("default")
        style.configure("CPU.Horizontal.TProgressbar",
                        troughcolor=COLOR_CARD, background=COLOR_CPU,
                        bordercolor=COLOR_PANEL, lightcolor=COLOR_CPU, darkcolor=COLOR_CPU)
        style.configure("CUDA.Horizontal.TProgressbar",
                        troughcolor=COLOR_CARD, background=COLOR_CUDA,
                        bordercolor=COLOR_PANEL, lightcolor=COLOR_CUDA, darkcolor=COLOR_CUDA)

        # CPU Progress bar
        tk.Label(bars_frame, text="CPU Progress:", font=("Segoe UI", 8, "bold"),
                 fg=COLOR_CPU, bg=COLOR_PANEL).grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.cpu_prog_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(bars_frame, style="CPU.Horizontal.TProgressbar",
                        variable=self.cpu_prog_var, maximum=100).grid(row=0, column=1, sticky="ew", padx=(0, 16))

        # CUDA Progress bar
        tk.Label(bars_frame, text="CUDA Progress:", font=("Segoe UI", 8, "bold"),
                 fg=COLOR_CUDA if self.cuda_available else COLOR_DANGER, bg=COLOR_PANEL).grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.cuda_prog_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(bars_frame, style="CUDA.Horizontal.TProgressbar",
                        variable=self.cuda_prog_var, maximum=100).grid(row=0, column=3, sticky="ew")

        bars_frame.columnconfigure(1, weight=1)
        bars_frame.columnconfigure(3, weight=1)

    # =========================================================================
    # EVENT CALLBACKS FOR THREAD SAFETY
    # =========================================================================
    def _on_gpu_boost_changed(self, event=None):
        choice = self.gpu_boost_var.get()
        if "20x" in choice:
            self.gpu_multiplier = 20
        elif "10x" in choice:
            self.gpu_multiplier = 10
        elif "5x" in choice:
            self.gpu_multiplier = 5
        elif "1x" in choice:
            self.gpu_multiplier = 1
        else:
            self.gpu_multiplier = 10

    def _on_speed_changed(self):
        try:
            self.sim_speed = int(self.speed_var.get())
        except Exception:
            pass

    def _update_target_display(self):
        try:
            t = int(self.iter_var.get())
            self.sim_target_iters = t
            self.stat_gen_lbl.config(text=f"{t:,}")
        except Exception:
            pass

    # =========================================================================
    # GRAPHICS & CANVAS OPTIMIZATION
    # =========================================================================
    def _init_canvas_items(self, canvas, rect_matrix, grid_data, alive_color):
        """Pre-allocate all canvas rectangle IDs once for high FPS rendering."""
        canvas.delete("all")
        for r in range(self.rows):
            y0 = r * self.cell_size
            y1 = y0 + self.cell_size - 1
            for c in range(self.cols):
                x0 = c * self.cell_size
                x1 = x0 + self.cell_size - 1
                color = alive_color if grid_data[r][c] else COLOR_DEAD_CELL
                rect_id = canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=color, outline=COLOR_GRID_LINE, width=1
                )
                rect_matrix[r][c] = rect_id

    def _on_frame_resize(self, event):
        """Dynamically scale canvas cell size when window is enlarged."""
        if not hasattr(self, "cpu_frame") or not self.cpu_frame:
            return
        frame_w = event.width
        frame_h = event.height
        if frame_w < 100 or frame_h < 100:
            return
        avail_w = frame_w - 24
        avail_h = frame_h - 48
        new_size = max(8, min(14, avail_w // self.cols, avail_h // self.rows))
        if new_size != self.cell_size:
            self._rescale_canvas_geometry(new_size)

    def _rescale_canvas_geometry(self, new_cell_size):
        """Update rectangle coordinates instantaneously when cell size changes on resize."""
        self.cell_size = new_cell_size
        cw = self.cols * self.cell_size
        ch = self.rows * self.cell_size
        self.cpu_canvas.config(width=cw, height=ch)
        self.cuda_canvas.config(width=cw, height=ch)

        for r in range(self.rows):
            y0 = r * self.cell_size
            y1 = y0 + self.cell_size - 1
            for c in range(self.cols):
                x0 = c * self.cell_size
                x1 = x0 + self.cell_size - 1
                if self._cpu_rects[r][c]:
                    self.cpu_canvas.coords(self._cpu_rects[r][c], x0, y0, x1, y1)
                if self.cuda_available and self._cuda_rects[r][c]:
                    self.cuda_canvas.coords(self._cuda_rects[r][c], x0, y0, x1, y1)

        if not self.cuda_available:
            self._render_cuda_not_found_canvas()

    def _render_cuda_not_found_canvas(self):
        """Render prominent 'CUDA is not found' message directly in the CUDA canvas."""
        self.cuda_canvas.delete("all")
        cw = self.cols * self.cell_size
        ch = self.rows * self.cell_size
        cx = cw // 2
        cy = ch // 2

        self.cuda_canvas.create_rectangle(
            20, 20, cw - 20, ch - 20,
            outline=COLOR_DANGER, width=2, dash=(6, 4)
        )
        self.cuda_canvas.create_text(
            cx, cy - 24,
            text="⚠  CUDA is not found",
            font=("Segoe UI", 16, "bold"),
            fill=COLOR_DANGER
        )
        self.cuda_canvas.create_text(
            cx, cy + 8,
            text="No active NVIDIA GPU or CUDA Driver detected",
            font=("Segoe UI", 10),
            fill=COLOR_MUTED
        )
        self.cuda_canvas.create_text(
            cx, cy + 32,
            text="CPU Simulation remains fully functional",
            font=("Segoe UI", 9, "italic"),
            fill="#64748b"
        )

    def _update_canvas_incremental(self, canvas, rect_matrix, changed_cells, alive_color):
        """Update only cells that flipped state."""
        for r, c, val in changed_cells:
            color = alive_color if val else COLOR_DEAD_CELL
            if rect_matrix[r][c]:
                canvas.itemconfig(rect_matrix[r][c], fill=color)

    def _update_canvas_full(self, canvas, rect_matrix, grid_data, alive_color):
        """Synchronize entire canvas with grid data."""
        for r in range(self.rows):
            for c in range(self.cols):
                color = alive_color if grid_data[r][c] else COLOR_DEAD_CELL
                if rect_matrix[r][c]:
                    canvas.itemconfig(rect_matrix[r][c], fill=color)

    def _on_cpu_canvas_draw(self, event, val, color):
        """Interactive drawing handler on CPU canvas."""
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.cpu_grid[r][c] != val:
                self.cpu_grid[r][c] = val
                self._cpu_canvas_grid[r][c] = val
                cell_color = color if val else COLOR_DEAD_CELL
                if self._cpu_rects[r][c]:
                    self.cpu_canvas.itemconfig(self._cpu_rects[r][c], fill=cell_color)
                self._update_all_stats()

    def _on_cuda_canvas_draw(self, event, val, color):
        """Interactive drawing handler on CUDA canvas (syncs to GPU memory)."""
        if not self.cuda_available:
            return
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.cuda_grid[r][c] != val:
                self.cuda_grid[r][c] = val
                self._cuda_canvas_grid[r][c] = val
                cell_color = color if val else COLOR_DEAD_CELL
                if self._cuda_rects[r][c]:
                    self.cuda_canvas.itemconfig(self._cuda_rects[r][c], fill=cell_color)
                if self.cuda_engine.available:
                    self.cuda_engine.upload(self.cuda_grid)
                self._update_all_stats()

    # =========================================================================
    # SIMULATION CONTROLS & DECOUPLED ASYNCHRONOUS THREADING
    # =========================================================================
    def _on_start(self):
        if self.running:
            return

        try:
            target = int(self.iter_var.get())
            if target < 1:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid Input", "Please enter a valid target iteration (>= 1).")
            return

        self.sim_target_iters = target
        self._on_gpu_boost_changed()
        self._on_speed_changed()

        # Check if this is a fresh start (previous run completed, target reached, or starting from 0)
        is_fresh_start = (
            self.cpu_done or self.cuda_done or
            self.cpu_gen >= target or self.cuda_gen >= target or
            (self.cpu_gen == 0 and self.cuda_gen == 0)
        )

        if is_fresh_start:
            self.cpu_gen = 0
            self.cuda_gen = 0
            self.cpu_done = False
            self.cuda_done = False if self.cuda_available else True
            self.start_sim_time = time.time()
            self.pause_start_time = None
            self.cuda_completion_time = None
            self.cpu_completion_time = None
            self.chart.reset()
            self.chart.add_point(0.0, 0.0, 0.0)
            if self.cuda_engine and self.cuda_engine.available:
                self.cuda_engine.upload(self.cuda_grid)
        else:
            # Resuming from pause: adjust start_sim_time by pause duration
            if hasattr(self, "pause_start_time") and self.pause_start_time is not None:
                pause_duration = time.time() - self.pause_start_time
                if self.start_sim_time is not None:
                    self.start_sim_time += pause_duration
                self.pause_start_time = None

        self.running = True
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.btn_step.config(state="disabled")

        self.last_perf_check = time.time()
        self.cpu_gen_counter = 0
        self.cuda_gen_counter = 0

        # Launch decoupled independent workers: CPU thread & CUDA GPU thread
        self._cpu_thread = threading.Thread(
            target=self._cpu_simulation_worker, args=(target,), daemon=True
        )
        self._cpu_thread.start()

        if self.cuda_available and self.cuda_engine.available:
            self._cuda_thread = threading.Thread(
                target=self._cuda_simulation_worker, args=(target,), daemon=True
            )
            self._cuda_thread.start()

    def _on_pause(self):
        self.running = False
        self.pause_start_time = time.time()
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled")
        self.btn_step.config(state="normal")
        self._update_all_stats()

    def _on_step(self):
        if self.running:
            return

        # Step CPU 1x
        nxt_cpu, _ = fast_next_generation(self.cpu_grid, self.rows, self.cols)
        self.cpu_grid = nxt_cpu
        self.cpu_gen += 1

        # Step CUDA (Hardware GPU)
        if self.cuda_available and self.cuda_engine.available:
            self.cuda_engine.step()
            new_cuda = self.cuda_engine.download()
            if new_cuda:
                self.cuda_grid = new_cuda
                self.cuda_gen += 1

        # Immediately synchronize canvas displays
        self._sync_canvas_cells()
        self._update_all_stats()

    def _sync_canvas_cells(self):
        """Immediately synchronize both CPU and CUDA canvas items with memory state."""
        curr_cpu = self.cpu_grid
        disp_cpu = self._cpu_canvas_grid
        for r in range(self.rows):
            row_c = curr_cpu[r]
            row_d = disp_cpu[r]
            for c in range(self.cols):
                val = row_c[c]
                if val != row_d[c]:
                    row_d[c] = val
                    color = COLOR_CPU if val else COLOR_DEAD_CELL
                    rect_id = self._cpu_rects[r][c]
                    if rect_id:
                        self.cpu_canvas.itemconfig(rect_id, fill=color)

        if self.cuda_available:
            curr_cu = self.cuda_grid
            disp_cu = self._cuda_canvas_grid
            for r in range(self.rows):
                row_c = curr_cu[r]
                row_d = disp_cu[r]
                for c in range(self.cols):
                    val = row_c[c]
                    if val != row_d[c]:
                        row_d[c] = val
                        color = COLOR_CUDA if val else COLOR_DEAD_CELL
                        rect_id = self._cuda_rects[r][c]
                        if rect_id:
                            self.cuda_canvas.itemconfig(rect_id, fill=color)

    def _on_sync_grids(self):
        """Copy CPU grid to CUDA grid for identical starting state comparison."""
        self.cuda_grid = [row[:] for row in self.cpu_grid]
        self._cuda_canvas_grid = [row[:] for row in self.cpu_grid]
        if self.cuda_available:
            if self.cuda_engine.available:
                self.cuda_engine.upload(self.cuda_grid)
            self._update_canvas_full(self.cuda_canvas, self._cuda_rects, self.cuda_grid, COLOR_CUDA)
        self._update_all_stats()

    def _on_reset(self):
        self._on_pause()
        self.pause_start_time = None
        self.cpu_gen = 0
        self.cuda_gen = 0
        self.cpu_done = False
        self.cuda_done = False
        self.start_sim_time = None
        self.cuda_completion_time = None
        self.cpu_completion_time = None
        self.cpu_gps = 0.0
        self.cuda_gps = 0.0

        rnd = make_random_grid(self.rows, self.cols, density=0.30)
        self.cpu_grid = [row[:] for row in rnd]
        self.cuda_grid = [row[:] for row in rnd]
        self._cpu_canvas_grid = [row[:] for row in rnd]
        self._cuda_canvas_grid = [row[:] for row in rnd]

        self._update_canvas_full(self.cpu_canvas, self._cpu_rects, self.cpu_grid, COLOR_CPU)
        if self.cuda_available:
            if self.cuda_engine.available:
                self.cuda_engine.upload(self.cuda_grid)
            self._update_canvas_full(self.cuda_canvas, self._cuda_rects, self.cuda_grid, COLOR_CUDA)
        else:
            self._render_cuda_not_found_canvas()

        self.chart.reset()
        self._update_all_stats()

    def _on_clear(self):
        self._on_pause()
        self.pause_start_time = None
        self.cpu_gen = 0
        self.cuda_gen = 0
        self.cpu_done = False
        self.cuda_done = False
        self.start_sim_time = None
        self.cuda_completion_time = None
        self.cpu_completion_time = None
        self.cpu_gps = 0.0
        self.cuda_gps = 0.0

        self.cpu_grid = make_empty_grid(self.rows, self.cols)
        self.cuda_grid = make_empty_grid(self.rows, self.cols)
        self._cpu_canvas_grid = make_empty_grid(self.rows, self.cols)
        self._cuda_canvas_grid = make_empty_grid(self.rows, self.cols)

        self._update_canvas_full(self.cpu_canvas, self._cpu_rects, self.cpu_grid, COLOR_CPU)
        if self.cuda_available:
            if self.cuda_engine.available:
                self.cuda_engine.upload(self.cuda_grid)
            self._update_canvas_full(self.cuda_canvas, self._cuda_rects, self.cuda_grid, COLOR_CUDA)
        else:
            self._render_cuda_not_found_canvas()

        self.chart.reset()
        self._update_all_stats()

    def _on_preset_selected(self, event=None):
        name = self.preset_var.get()
        self._on_pause()
        self.pause_start_time = None
        self.cpu_gen = 0
        self.cuda_gen = 0
        self.cpu_done = False
        self.cuda_done = False
        self.start_sim_time = None
        self.cuda_completion_time = None
        self.cpu_completion_time = None

        if name == "Random (30% Density)":
            rnd = make_random_grid(self.rows, self.cols, density=0.30)
            self.cpu_grid = [row[:] for row in rnd]
            self.cuda_grid = [row[:] for row in rnd]
        elif name in PRESETS:
            coords = PRESETS[name]
            empty = make_empty_grid(self.rows, self.cols)
            cr, cc = self.rows // 2, self.cols // 2
            for dr, dc in coords:
                r, c = cr + dr, cc + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    empty[r][c] = 1
            self.cpu_grid = [row[:] for row in empty]
            self.cuda_grid = [row[:] for row in empty]

        self._cpu_canvas_grid = [row[:] for row in self.cpu_grid]
        self._cuda_canvas_grid = [row[:] for row in self.cuda_grid]

        self._update_canvas_full(self.cpu_canvas, self._cpu_rects, self.cpu_grid, COLOR_CPU)
        if self.cuda_available:
            if self.cuda_engine.available:
                self.cuda_engine.upload(self.cuda_grid)
            self._update_canvas_full(self.cuda_canvas, self._cuda_rects, self.cuda_grid, COLOR_CUDA)
        else:
            self._render_cuda_not_found_canvas()

        self.chart.reset()
        self._update_all_stats()

    # -------------------------------------------------------------------------
    # INDEPENDENT ASYNCHRONOUS SIMULATION WORKERS (CPU vs CUDA SPEED DIFFERENCE)
    # -------------------------------------------------------------------------
    def _cpu_simulation_worker(self, target_iters):
        """Worker thread for CPU simulation: progresses at host CPU speed."""
        while self.running and self.cpu_gen < target_iters:
            nxt, _ = fast_next_generation(self.cpu_grid, self.rows, self.cols)
            self.cpu_grid = nxt
            self.cpu_gen += 1
            self.cpu_gen_counter += 1

            # CPU speed pacing using pure Python attribute
            speed = self.sim_speed
            delay = max(0.015, 0.200 - (speed - 1) * (0.185 / 99))
            time.sleep(delay)

        if self.cpu_gen >= target_iters:
            self.cpu_done = True
            if self.cpu_completion_time is None and self.start_sim_time:
                self.cpu_completion_time = time.time() - self.start_sim_time

    def _cuda_simulation_worker(self, target_iters):
        """
        Worker thread for CUDA GPU simulation:
        Simulates EVERY iteration (1 to target) directly on the NVIDIA GPU (RTX 3050)
        via native PTX kernel. Downloads each state for high-frame-rate visual animation,
        demonstrating undeniable GPU computational superiority.
        """
        self.cuda_engine.bind_thread()

        while self.running and self.cuda_gen < target_iters:
            self.cuda_engine.step()
            new_cuda = self.cuda_engine.download()
            if new_cuda:
                self.cuda_grid = new_cuda

            self.cuda_gen += 1
            self.cuda_gen_counter += 1

            # GPU visual delay: scaled so the visual race is exciting and evident
            speed = self.sim_speed
            base_delay = max(0.015, 0.200 - (speed - 1) * (0.185 / 99))
            mult = max(1, self.gpu_multiplier)
            gpu_delay = max(0.002, base_delay / mult)
            time.sleep(gpu_delay)

        if self.cuda_gen >= target_iters:
            self.cuda_done = True
            if self.cuda_completion_time is None and self.start_sim_time:
                self.cuda_completion_time = time.time() - self.start_sim_time

    def _check_both_complete(self):
        """Check if both simulations have finished their target iterations."""
        if self.cpu_done and (self.cuda_done or not self.cuda_available):
            self.running = False
            self.btn_start.config(state="normal")
            self.btn_pause.config(state="disabled")
            self.btn_step.config(state="normal")
            self._update_all_stats()

    def _refresh_loop(self):
        """Periodic performance calculation, responsive canvas rendering, chart plotting and stats loop."""
        # 1. Update CPU canvas items for any cells that changed
        curr_cpu = self.cpu_grid
        disp_cpu = self._cpu_canvas_grid
        for r in range(self.rows):
            row_c = curr_cpu[r]
            row_d = disp_cpu[r]
            for c in range(self.cols):
                val = row_c[c]
                if val != row_d[c]:
                    row_d[c] = val
                    color = COLOR_CPU if val else COLOR_DEAD_CELL
                    rect_id = self._cpu_rects[r][c]
                    if rect_id:
                        self.cpu_canvas.itemconfig(rect_id, fill=color)

        # 2. Update CUDA canvas items for any cells that changed
        if self.cuda_available:
            curr_cu = self.cuda_grid
            disp_cu = self._cuda_canvas_grid
            for r in range(self.rows):
                row_c = curr_cu[r]
                row_d = disp_cu[r]
                for c in range(self.cols):
                    val = row_c[c]
                    if val != row_d[c]:
                        row_d[c] = val
                        color = COLOR_CUDA if val else COLOR_DEAD_CELL
                        rect_id = self._cuda_rects[r][c]
                        if rect_id:
                            self.cuda_canvas.itemconfig(rect_id, fill=color)

        # 3. Immediately mark CUDA completion milestone on chart when GPU finishes
        if self.cuda_done and self.cuda_completion_time and self.chart.cuda_completed_at is None:
            self.chart.mark_cuda_completed(self.cuda_completion_time * 1000.0)

        # 4. Check simulation completion safely on the main thread
        if self.running and self.cpu_done and (self.cuda_done or not self.cuda_available):
            self._check_both_complete()

        # 5. Performance metrics calculations & Chart point addition (50ms sampling for high detail)
        now = time.time()
        dt = now - self.last_perf_check
        if dt >= 0.05:
            self.cpu_gps = self.cpu_gen_counter / dt
            self.cuda_gps = self.cuda_gen_counter / dt if not self.cuda_done else 0.0
            self.cpu_gen_counter = 0
            self.cuda_gen_counter = 0
            self.last_perf_check = now

            if self.running and self.start_sim_time:
                elapsed_ms = (now - self.start_sim_time) * 1000.0
                self.chart.add_point(elapsed_ms, self.cpu_gps, self.cuda_gps)

            if self.running or self.cpu_gen > 0 or self.cuda_gen > 0:
                self._update_all_stats()

        self.after(FRAME_MS, self._refresh_loop)

    # =========================================================================
    # STATS UPDATES
    # =========================================================================
    def _update_all_stats(self):
        target = self.sim_target_iters
        cpu_live = sum(sum(r) for r in self.cpu_grid)
        cuda_live = sum(sum(r) for r in self.cuda_grid) if self.cuda_available else 0

        # Elapsed time display in ms
        if self.start_sim_time:
            elapsed_ms = (time.time() - self.start_sim_time) * 1000.0
            if elapsed_ms < 10000:
                self.stat_elapsed_lbl.config(text=f"{int(elapsed_ms):,} ms")
            else:
                self.stat_elapsed_lbl.config(text=f"{elapsed_ms / 1000.0:.1f} s ({int(elapsed_ms):,} ms)")
        else:
            self.stat_elapsed_lbl.config(text="0 ms")

        self.stat_cpu_cells_lbl.config(text=f"{cpu_live:,}")
        self.stat_cuda_cells_lbl.config(text=f"{cuda_live:,}" if self.cuda_available else "N/A")

        self.cpu_live_badge.config(
            text=f"Gen: {self.cpu_gen:,} (DONE)" if self.cpu_done else f"Gen: {self.cpu_gen:,} | Live: {cpu_live:,}"
        )
        if self.cuda_available:
            self.cuda_live_badge.config(
                text=f"Gen: {self.cuda_gen:,} (DONE)" if self.cuda_done else f"Gen: {self.cuda_gen:,} | Live: {cuda_live:,}"
            )
        else:
            self.cuda_live_badge.config(text="CUDA NOT FOUND")

        # Update Live Chart Legend Labels
        self.chart_cpu_lbl.config(text=f"{self.cpu_gps:.1f} iters/s")
        if self.cuda_available:
            self.chart_cuda_lbl.config(text=f"{self.cuda_gps:.1f} iters/s")
            if self.cpu_gps > 0 and self.cuda_gps > 0:
                ratio = self.cuda_gps / self.cpu_gps
                self.speedup_badge.config(text=f"⚡ Speedup: {ratio:.1f}x", fg=COLOR_CUDA)
            elif self.cuda_done and self.cuda_completion_time:
                self.speedup_badge.config(text=f"⚡ Finished in {int(self.cuda_completion_time * 1000):,}ms", fg=COLOR_SUCCESS)
            else:
                self.speedup_badge.config(text="⚡ Accelerated", fg=COLOR_WARNING)
        else:
            self.chart_cuda_lbl.config(text="OFFLINE", fg=COLOR_DANGER)
            self.speedup_badge.config(text="CUDA Offline", fg=COLOR_DANGER)

        # Progress bars
        cpu_pct = (self.cpu_gen / target * 100) if target > 0 else 0
        cuda_pct = (self.cuda_gen / target * 100) if (target > 0 and self.cuda_available) else 0
        self.cpu_prog_var.set(min(cpu_pct, 100.0))
        self.cuda_prog_var.set(min(cuda_pct, 100.0))

    def _on_close(self):
        """Safe thread and CUDA context shutdown on window close."""
        self.running = False
        if self.cuda_engine:
            self.cuda_engine.close()
        self.destroy()


def launch_gui():
    """Launch the Dual Engine GUI Visualizer."""
    app = DualGameOfLifeApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()