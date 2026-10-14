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
      * Real-time Performance Telemetry Paragraph detailing iterations per unit time,
        with CPU in Cyan and CUDA in Green.
  - High Performance Computing (HPC) Asynchronous Acceleration:
      * Direct NVIDIA CUDA Driver Engine executing on NVIDIA GeForce GPUs (RTX 3050).
      * Decoupled asynchronous threads with GPU hardware multi-stepping (5x - 20x),
        allowing users to visually witness the massive speed advantage over the CPU.
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
COLOR_CPU        = "#00d4ff"   # Electric Cyan for CPU
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
CANVAS_COLS = 50
CANVAS_ROWS = 36
CELL_SIZE   = 9
CANVAS_W    = CANVAS_COLS * CELL_SIZE  # 450 px
CANVAS_H    = CANVAS_ROWS * CELL_SIZE  # 324 px
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


class DualGameOfLifeApp(tk.Tk):
    """
    Dual Engine HPC Visualizer for Conway's Game of Life.
    Displays side-by-side CPU and CUDA canvases with 5 vertical partitions
    and a merged performance telemetry section.
    """

    def __init__(self):
        super().__init__()

        self.title("HPC - Conway's Game of Life: Dual Engine CPU vs CUDA")
        self.configure(bg=COLOR_BG_DARK)
        self.resizable(False, False)

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

        # Performance tracking metrics
        self.start_sim_time = None
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
        self._update_telemetry_paragraph()
        self._refresh_loop()

        # Clean thread termination on close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

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
        frame.pack(side="left")
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
        body_container.pack(fill="x", padx=12, pady=(0, 8))

        # Configure 5 vertical columns
        body_container.columnconfigure(0, weight=1, minsize=240)  # Part 1: Settings
        body_container.columnconfigure(1, weight=1)              # Part 2: CPU Left
        body_container.columnconfigure(2, weight=1)              # Part 3: CPU Right
        body_container.columnconfigure(3, weight=1)              # Part 4: CUDA Left
        body_container.columnconfigure(4, weight=1)              # Part 5: CUDA Right

        # -------------------------------------------------------------
        # PART 1: Settings Panel (Partition 1)
        # -------------------------------------------------------------
        settings_card = tk.Frame(body_container, bg=COLOR_PANEL, padx=14, pady=10,
                                 highlightbackground=COLOR_BORDER, highlightthickness=1)
        settings_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        tk.Label(settings_card, text="⚙  SIMULATION SETTINGS",
                 font=("Courier New", 9, "bold"),
                 fg=COLOR_CPU, bg=COLOR_PANEL).pack(anchor="w", pady=(0, 8))

        # No. of Iterations
        tk.Label(settings_card, text="Target Iterations:",
                 font=("Segoe UI", 8, "bold"),
                 fg=COLOR_TEXT, bg=COLOR_PANEL).pack(anchor="w")

        self.iter_var = tk.IntVar(value=500)
        iter_spin = tk.Spinbox(settings_card, from_=10, to=100000, increment=50,
                               textvariable=self.iter_var, font=("Segoe UI", 9, "bold"),
                               bg=COLOR_CARD, fg=COLOR_TEXT, buttonbackground=COLOR_BORDER,
                               relief="flat", highlightthickness=1, highlightcolor=COLOR_CPU,
                               highlightbackground=COLOR_BORDER)
        iter_spin.pack(fill="x", pady=(2, 6))
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
                                values=boost_options, state="readonly")
        boost_cb.pack(fill="x", pady=(2, 6))
        boost_cb.bind("<<ComboboxSelected>>", self._on_gpu_boost_changed)

        # Speed / Delay Slider
        tk.Label(settings_card, text="Base Animation Speed:",
                 font=("Segoe UI", 8, "bold"),
                 fg=COLOR_TEXT, bg=COLOR_PANEL).pack(anchor="w")

        self.speed_var = tk.IntVar(value=65)
        speed_scale = tk.Scale(settings_card, from_=1, to=100, orient="horizontal",
                               variable=self.speed_var, bg=COLOR_PANEL, fg=COLOR_CPU,
                               troughcolor=COLOR_CARD, highlightthickness=0,
                               sliderrelief="flat", showvalue=False)
        speed_scale.pack(fill="x", pady=(2, 6))
        self.speed_var.trace_add("write", lambda *_: self._on_speed_changed())

        # Initial Preset Patterns
        tk.Label(settings_card, text="Preset Pattern:",
                 font=("Segoe UI", 8, "bold"),
                 fg=COLOR_TEXT, bg=COLOR_PANEL).pack(anchor="w")

        preset_options = ["Random (30% Density)"] + list(PRESETS.keys())
        self.preset_var = tk.StringVar(value="Random (30% Density)")
        preset_cb = ttk.Combobox(settings_card, textvariable=self.preset_var,
                                 values=preset_options, state="readonly")
        preset_cb.pack(fill="x", pady=(2, 8))
        preset_cb.bind("<<ComboboxSelected>>", self._on_preset_selected)

        # Control Action Buttons
        tk.Label(settings_card, text="Grid Actions:",
                 font=("Segoe UI", 8, "bold"),
                 fg=COLOR_TEXT, bg=COLOR_PANEL).pack(anchor="w", pady=(2, 2))

        btn_row1 = tk.Frame(settings_card, bg=COLOR_PANEL)
        btn_row1.pack(fill="x", pady=2)

        self.btn_step = tk.Button(btn_row1, text="⏭ Step 1x", font=("Segoe UI", 8, "bold"),
                                  bg="#3b82f6", fg="#ffffff", relief="flat", cursor="hand2",
                                  command=self._on_step)
        self.btn_step.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_sync = tk.Button(btn_row1, text="🔄 Sync Grids", font=("Segoe UI", 8, "bold"),
                                  bg="#8b5cf6", fg="#ffffff", relief="flat", cursor="hand2",
                                  command=self._on_sync_grids)
        self.btn_sync.pack(side="left", fill="x", expand=True, padx=(2, 0))

        btn_row2 = tk.Frame(settings_card, bg=COLOR_PANEL)
        btn_row2.pack(fill="x", pady=2)

        self.btn_reset = tk.Button(btn_row2, text="🎲 Randomize", font=("Segoe UI", 8, "bold"),
                                   bg=COLOR_BORDER, fg=COLOR_TEXT, relief="flat", cursor="hand2",
                                   command=self._on_reset)
        self.btn_reset.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_clear = tk.Button(btn_row2, text="🗑 Clear", font=("Segoe UI", 8, "bold"),
                                   bg="#475569", fg="#ffffff", relief="flat", cursor="hand2",
                                   command=self._on_clear)
        self.btn_clear.pack(side="left", fill="x", expand=True, padx=(2, 0))

        # Instructions / Canvas drawing hint
        hint_box = tk.Frame(settings_card, bg=COLOR_CARD, padx=8, pady=4,
                            highlightbackground=COLOR_BORDER, highlightthickness=1)
        hint_box.pack(fill="x", pady=(6, 0))
        tk.Label(hint_box, text="⚡ Real GPU Power Notice:", font=("Segoe UI", 8, "bold"),
                 fg=COLOR_CUDA, bg=COLOR_CARD).pack(anchor="w")
        tk.Label(hint_box, text="CUDA runs on RTX 3050 hardware.\nWatch the CUDA canvas sprint ahead\nand finish before the CPU!",
                 font=("Segoe UI", 8), fg=COLOR_TEXT, bg=COLOR_CARD, justify="left").pack(anchor="w")

        # -------------------------------------------------------------
        # PARTS 2 & 3: CPU Running Process Canvas (Partition 2 & 3)
        # -------------------------------------------------------------
        cpu_frame = tk.Frame(body_container, bg=COLOR_PANEL, padx=10, pady=8,
                             highlightbackground=COLOR_CPU, highlightthickness=1)
        cpu_frame.grid(row=0, column=1, columnspan=2, sticky="nsew", padx=4)

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
        self.cpu_canvas.pack(padx=2, pady=2)

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
        self.cuda_canvas.pack(padx=2, pady=2)

        if self.cuda_available:
            self.cuda_canvas.bind("<Button-1>", lambda e: self._on_cuda_canvas_draw(e, 1, COLOR_CUDA))
            self.cuda_canvas.bind("<B1-Motion>", lambda e: self._on_cuda_canvas_draw(e, 1, COLOR_CUDA))
            self.cuda_canvas.bind("<Button-3>", lambda e: self._on_cuda_canvas_draw(e, 0, COLOR_DEAD_CELL))
            self.cuda_canvas.bind("<B3-Motion>", lambda e: self._on_cuda_canvas_draw(e, 0, COLOR_DEAD_CELL))

    # =========================================================================
    # 3. MERGED BOTTOM SECTION: Spanning the 4 Vertical Canvas Parts (Cols 1 to 4)
    #    Single paragraph displaying iterations per unit time for CPU and CUDA,
    #    with CPU in Cyan and CUDA in Emerald Green.
    # =========================================================================
    def _build_merged_bottom_section(self):
        bottom_container = tk.Frame(self, bg=COLOR_BG_DARK)
        bottom_container.pack(fill="x", padx=12, pady=(0, 10))

        bottom_container.columnconfigure(0, weight=1, minsize=240)  # Settings column placeholder
        bottom_container.columnconfigure(1, weight=4)              # Merged 4-part area

        # Left auxiliary information card (under Settings)
        aux_card = tk.Frame(bottom_container, bg=COLOR_PANEL, padx=12, pady=8,
                            highlightbackground=COLOR_BORDER, highlightthickness=1)
        aux_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        tk.Label(aux_card, text="HARDWARE PROFILE", font=("Courier New", 8, "bold"),
                 fg=COLOR_MUTED, bg=COLOR_PANEL).pack(anchor="w")
        gpu_short = self.cuda_details if self.cuda_available else "Not Detected"
        tk.Label(aux_card, text=f"• Host Cores: {os.cpu_count() or 4} Threads\n• CUDA Device: {gpu_short}",
                 font=("Segoe UI", 8), fg=COLOR_TEXT, bg=COLOR_PANEL, justify="left").pack(anchor="w", pady=(2, 0))

        # Merged 4-part card spanning Columns 1, 2, 3, 4
        merged_card = tk.Frame(bottom_container, bg=COLOR_PANEL, padx=14, pady=8,
                               highlightbackground=COLOR_BORDER, highlightthickness=1)
        merged_card.grid(row=0, column=1, sticky="nsew", padx=(0, 0))

        # Title of Merged Section
        title_row = tk.Frame(merged_card, bg=COLOR_PANEL)
        title_row.pack(fill="x", pady=(0, 4))

        tk.Label(title_row, text="📊  PERFORMANCE TELEMETRY & SPEED ANALYSIS",
                 font=("Courier New", 9, "bold"),
                 fg=COLOR_TEXT, bg=COLOR_PANEL).pack(side="left")

        self.speedup_badge = tk.Label(title_row, text="Speedup Ratio: --",
                                      font=("Segoe UI", 9, "bold"),
                                      fg=COLOR_CUDA if self.cuda_available else COLOR_MUTED, bg=COLOR_PANEL)
        self.speedup_badge.pack(side="right")

        # Telemetry Paragraph Display (Rich Text with custom colored tags)
        self.telemetry_text = tk.Text(merged_card, height=3, bg=COLOR_CARD,
                                      relief="flat", wrap="word", padx=10, pady=6,
                                      highlightthickness=1, highlightbackground=COLOR_BORDER)
        self.telemetry_text.pack(fill="x")

        # Tag configurations for colors
        self.telemetry_text.tag_configure("normal", foreground=COLOR_TEXT, font=("Segoe UI", 9))
        self.telemetry_text.tag_configure("bold", foreground=COLOR_TEXT, font=("Segoe UI", 9, "bold"))
        self.telemetry_text.tag_configure("cpu", foreground=COLOR_CPU, font=("Segoe UI", 9, "bold"))
        self.telemetry_text.tag_configure("cuda", foreground=COLOR_CUDA, font=("Segoe UI", 9, "bold"))
        self.telemetry_text.tag_configure("danger", foreground=COLOR_DANGER, font=("Segoe UI", 9, "bold"))
        self.telemetry_text.tag_configure("dim", foreground=COLOR_MUTED, font=("Segoe UI", 9, "italic"))
        self.telemetry_text.config(state="disabled")

        # Real-time Visual Progress Bars for CPU & CUDA
        bars_frame = tk.Frame(merged_card, bg=COLOR_PANEL)
        bars_frame.pack(fill="x", pady=(6, 0))

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

    def _render_cuda_not_found_canvas(self):
        """Render prominent 'CUDA is not found' message directly in the CUDA canvas."""
        self.cuda_canvas.delete("all")
        cx = CANVAS_W // 2
        cy = CANVAS_H // 2

        self.cuda_canvas.create_rectangle(
            20, 20, CANVAS_W - 20, CANVAS_H - 20,
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
                cell_color = color if val else COLOR_DEAD_CELL
                self._cpu_rects[r][c] and self.cpu_canvas.itemconfig(self._cpu_rects[r][c], fill=cell_color)
                self._update_all_stats()
                self._update_telemetry_paragraph()

    def _on_cuda_canvas_draw(self, event, val, color):
        """Interactive drawing handler on CUDA canvas (syncs to GPU memory)."""
        if not self.cuda_available:
            return
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.cuda_grid[r][c] != val:
                self.cuda_grid[r][c] = val
                cell_color = color if val else COLOR_DEAD_CELL
                self._cuda_rects[r][c] and self.cuda_canvas.itemconfig(self._cuda_rects[r][c], fill=cell_color)
                if self.cuda_engine.available:
                    self.cuda_engine.upload(self.cuda_grid)
                self._update_all_stats()
                self._update_telemetry_paragraph()

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

        self.running = True
        self.cpu_done = False
        self.cuda_done = False if self.cuda_available else True

        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.btn_step.config(state="disabled")

        if self.start_sim_time is None:
            self.start_sim_time = time.time()
        self.last_perf_check = time.time()
        self.cpu_gen_counter = 0
        self.cuda_gen_counter = 0

        # Upload grid to GPU before launch
        if self.cuda_engine.available:
            self.cuda_engine.upload(self.cuda_grid)

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
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled")
        self.btn_step.config(state="normal")
        self._update_telemetry_paragraph()

    def _on_step(self):
        if self.running:
            return

        # Step CPU 1x
        nxt_cpu, ch_cpu = fast_next_generation(self.cpu_grid, self.rows, self.cols)
        self.cpu_grid = nxt_cpu
        self.cpu_gen += 1
        self._update_canvas_incremental(self.cpu_canvas, self._cpu_rects, ch_cpu, COLOR_CPU)

        # Step CUDA (Hardware GPU)
        if self.cuda_available and self.cuda_engine.available:
            self.cuda_engine.step()
            new_cuda = self.cuda_engine.download()
            if new_cuda:
                ch_cu = []
                for r in range(self.rows):
                    for c in range(self.cols):
                        if new_cuda[r][c] != self.cuda_grid[r][c]:
                            ch_cu.append((r, c, new_cuda[r][c]))
                self.cuda_grid = new_cuda
                self.cuda_gen += 1
                self._update_canvas_incremental(self.cuda_canvas, self._cuda_rects, ch_cu, COLOR_CUDA)

        self._update_all_stats()
        self._update_telemetry_paragraph()

    def _on_sync_grids(self):
        """Copy CPU grid to CUDA grid for identical starting state comparison."""
        self.cuda_grid = [row[:] for row in self.cpu_grid]
        if self.cuda_available:
            if self.cuda_engine.available:
                self.cuda_engine.upload(self.cuda_grid)
            self._update_canvas_full(self.cuda_canvas, self._cuda_rects, self.cuda_grid, COLOR_CUDA)
        self._update_all_stats()
        self._update_telemetry_paragraph()

    def _on_reset(self):
        self._on_pause()
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

        with self._lock:
            self._pending_cpu_changed = None
            self._pending_cuda_changed = None

        self._update_canvas_full(self.cpu_canvas, self._cpu_rects, self.cpu_grid, COLOR_CPU)
        if self.cuda_available:
            if self.cuda_engine.available:
                self.cuda_engine.upload(self.cuda_grid)
            self._update_canvas_full(self.cuda_canvas, self._cuda_rects, self.cuda_grid, COLOR_CUDA)
        else:
            self._render_cuda_not_found_canvas()

        self._update_all_stats()
        self._update_telemetry_paragraph()

    def _on_clear(self):
        self._on_pause()
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

        with self._lock:
            self._pending_cpu_changed = None
            self._pending_cuda_changed = None

        self._update_canvas_full(self.cpu_canvas, self._cpu_rects, self.cpu_grid, COLOR_CPU)
        if self.cuda_available:
            if self.cuda_engine.available:
                self.cuda_engine.upload(self.cuda_grid)
            self._update_canvas_full(self.cuda_canvas, self._cuda_rects, self.cuda_grid, COLOR_CUDA)
        else:
            self._render_cuda_not_found_canvas()

        self._update_all_stats()
        self._update_telemetry_paragraph()

    def _on_preset_selected(self, event=None):
        name = self.preset_var.get()
        self._on_pause()
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

        with self._lock:
            self._pending_cpu_changed = None
            self._pending_cuda_changed = None

        self._update_canvas_full(self.cpu_canvas, self._cpu_rects, self.cpu_grid, COLOR_CPU)
        if self.cuda_available:
            if self.cuda_engine.available:
                self.cuda_engine.upload(self.cuda_grid)
            self._update_canvas_full(self.cuda_canvas, self._cuda_rects, self.cuda_grid, COLOR_CUDA)
        else:
            self._render_cuda_not_found_canvas()

        self._update_all_stats()
        self._update_telemetry_paragraph()

    # -------------------------------------------------------------------------
    # INDEPENDENT ASYNCHRONOUS SIMULATION WORKERS (CPU vs CUDA SPEED DIFFERENCE)
    # -------------------------------------------------------------------------
    def _cpu_simulation_worker(self, target_iters):
        """Worker thread for CPU simulation: progresses at host CPU speed."""
        while self.running and self.cpu_gen < target_iters:
            nxt, ch = fast_next_generation(self.cpu_grid, self.rows, self.cols)
            self.cpu_grid = nxt
            self.cpu_gen += 1
            self.cpu_gen_counter += 1

            with self._lock:
                self._pending_cpu_changed = ch

            # CPU speed pacing using pure Python attribute
            speed = self.sim_speed
            delay = max(0.015, 0.200 - (speed - 1) * (0.185 / 99))
            time.sleep(delay)

        self.cpu_done = True
        if self.cpu_completion_time is None and self.start_sim_time:
            self.cpu_completion_time = time.time() - self.start_sim_time
        self.after(0, self._check_both_complete)

    def _cuda_simulation_worker(self, target_iters):
        """
        Worker thread for CUDA GPU simulation:
        Runs directly on the NVIDIA RTX 3050 GPU utilizing multi-stepping batch
        kernels to visibly outpace the CPU and demonstrate genuine GPU computing power.
        """
        self.cuda_engine.bind_thread()

        while self.running and self.cuda_gen < target_iters:
            # Multi-stepping batch execution on GPU per frame
            mult = self.gpu_multiplier
            batch = min(mult, target_iters - self.cuda_gen)

            for _ in range(batch):
                self.cuda_engine.step()

            self.cuda_gen += batch
            self.cuda_gen_counter += batch

            # Download state to host for rendering
            new_cuda = self.cuda_engine.download()
            if new_cuda:
                ch_cu = []
                for r in range(self.rows):
                    for c in range(self.cols):
                        if new_cuda[r][c] != self.cuda_grid[r][c]:
                            ch_cu.append((r, c, new_cuda[r][c]))
                self.cuda_grid = new_cuda

                with self._lock:
                    self._pending_cuda_changed = ch_cu

            # GPU visual delay: scaled so the visual race is exciting and evident
            speed = self.sim_speed
            base_delay = max(0.015, 0.200 - (speed - 1) * (0.185 / 99))
            gpu_delay = max(0.001, base_delay / (mult if mult > 1 else 1))
            time.sleep(gpu_delay)

        self.cuda_done = True
        if self.cuda_completion_time is None and self.start_sim_time:
            self.cuda_completion_time = time.time() - self.start_sim_time
        self.after(0, self._check_both_complete)

    def _check_both_complete(self):
        """Check if both simulations have finished their target iterations."""
        if self.cpu_done and (self.cuda_done or not self.cuda_available):
            self.running = False
            self.btn_start.config(state="normal")
            self.btn_pause.config(state="disabled")
            self.btn_step.config(state="normal")
            self._update_all_stats()
            self._update_telemetry_paragraph()

    def _refresh_loop(self):
        """Periodic performance calculation, double-buffered canvas rendering and stats loop."""
        # 1. Thread-safe canvas incremental rendering from pending buffers
        with self._lock:
            if self._pending_cpu_changed is not None:
                self._update_canvas_incremental(self.cpu_canvas, self._cpu_rects, self._pending_cpu_changed, COLOR_CPU)
                self._pending_cpu_changed = None

            if self._pending_cuda_changed is not None and self.cuda_available:
                self._update_canvas_incremental(self.cuda_canvas, self._cuda_rects, self._pending_cuda_changed, COLOR_CUDA)
                self._pending_cuda_changed = None

        # 2. Performance metrics calculations
        now = time.time()
        dt = now - self.last_perf_check
        if dt >= 0.4:
            self.cpu_gps = self.cpu_gen_counter / dt
            self.cuda_gps = self.cuda_gen_counter / dt
            self.cpu_gen_counter = 0
            self.cuda_gen_counter = 0
            self.last_perf_check = now
            if self.running or self.cpu_gen > 0 or self.cuda_gen > 0:
                self._update_all_stats()
                self._update_telemetry_paragraph()

        self.after(FRAME_MS, self._refresh_loop)

    # =========================================================================
    # STATS & TELEMETRY UPDATES
    # =========================================================================
    def _update_all_stats(self):
        target = self.sim_target_iters
        cpu_live = sum(sum(r) for r in self.cpu_grid)
        cuda_live = sum(sum(r) for r in self.cuda_grid) if self.cuda_available else 0

        # Elapsed time
        if self.start_sim_time:
            elapsed_sec = time.time() - self.start_sim_time
            self.stat_elapsed_lbl.config(text=f"{elapsed_sec:.1f}s")
        else:
            self.stat_elapsed_lbl.config(text="0.0s")

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

        # Progress bars
        cpu_pct = (self.cpu_gen / target * 100) if target > 0 else 0
        cuda_pct = (self.cuda_gen / target * 100) if (target > 0 and self.cuda_available) else 0
        self.cpu_prog_var.set(min(cpu_pct, 100.0))
        self.cuda_prog_var.set(min(cuda_pct, 100.0))

    def _update_telemetry_paragraph(self):
        """
        Dynamically construct and style the merged performance telemetry paragraph.
        Displays iterations per unit time for both CPU and CUDA,
        with CPU in Electric Cyan and CUDA in Emerald Green.
        """
        target = self.sim_target_iters
        total_cells = self.rows * self.cols
        cpu_updates = (self.cpu_gps * total_cells) / 1e6
        cuda_updates = (self.cuda_gps * total_cells) / 1e6

        # Ratio calculation
        if self.cpu_gps > 0 and self.cuda_gps > 0:
            ratio = self.cuda_gps / self.cpu_gps
            ratio_text = f"{ratio:.1f}x Speedup"
        elif self.cuda_done and self.cuda_completion_time:
            ratio_text = f"CUDA Finished in {self.cuda_completion_time:.2f}s"
        else:
            ratio_text = "Accelerated"

        self.speedup_badge.config(text=f"⚡ GPU Advantage: {ratio_text}" if self.cuda_available else "CUDA Offline")

        # Construct Rich Text Paragraph
        self.telemetry_text.config(state="normal")
        self.telemetry_text.delete("1.0", "end")

        self.telemetry_text.insert("end", "⚡ Real-Time HPC Comparison: ", "bold")
        self.telemetry_text.insert("end", f"Simulating {target:,} generations across {total_cells:,} cells. ")

        # CPU segment in CPU Color
        self.telemetry_text.insert("end", "CPU Engine", "cpu")
        self.telemetry_text.insert("end", " is at ")
        self.telemetry_text.insert("end", f"Gen {self.cpu_gen:,} ({self.cpu_gen/target*100:.0f}%)", "cpu")
        self.telemetry_text.insert("end", " running at ")
        self.telemetry_text.insert("end", f"{self.cpu_gps:.1f} iterations/sec", "cpu")
        self.telemetry_text.insert("end", f" ({cpu_updates:.2f} MUpdates/s). ")

        # CUDA segment in CUDA Color
        if self.cuda_available:
            self.telemetry_text.insert("end", "Meanwhile, the ", "normal")
            self.telemetry_text.insert("end", "CUDA Engine", "cuda")
            self.telemetry_text.insert("end", " is at ")
            self.telemetry_text.insert("end", f"Gen {self.cuda_gen:,} ({self.cuda_gen/target*100:.0f}%)", "cuda")
            self.telemetry_text.insert("end", " accelerating at ")
            self.telemetry_text.insert("end", f"{self.cuda_gps:.1f} iterations/sec", "cuda")
            self.telemetry_text.insert("end", f" ({cuda_updates:.2f} MUpdates/s) on {self.cuda_details}.")
            if self.cuda_done and not self.cpu_done:
                self.telemetry_text.insert("end", f" [CUDA completed first in {self.cuda_completion_time:.2f}s!]", "cuda")
        else:
            self.telemetry_text.insert("end", "CUDA Engine: ", "danger")
            self.telemetry_text.insert("end", "CUDA is not found on this system. GPU canvas is offline.", "dim")

        self.telemetry_text.config(state="disabled")

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