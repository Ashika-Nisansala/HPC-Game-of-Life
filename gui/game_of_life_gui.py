"""
HPC Game of Life - GUI Visualizer & Simulation Engine
High-performance Conway's Game of Life simulation with dynamic Tkinter animation,
interactive canvas drawing, preset patterns, and benchmark capabilities.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import threading

# --- Visual Theme Palette ---
BG_DARK        = "#0d0f1a"
BG_PANEL       = "#111827"
BG_CARD        = "#1a2035"
ACCENT         = "#00d4ff"
ACCENT2        = "#7c3aed"
ALIVE_COLOR    = "#00d4ff"
DEAD_COLOR     = "#0d1526"
GRID_COLOR     = "#162032"
TEXT_PRIMARY   = "#e2e8f0"
TEXT_SECONDARY = "#94a3b8"
BTN_START      = "#00d4ff"
BTN_STOP       = "#ef4444"
BTN_RESET      = "#7c3aed"
BTN_STEP       = "#3b82f6"
BTN_CLEAR      = "#64748b"
BTN_TEXT       = "#0d0f1a"

# --- Simulation Parameters ---
DEFAULT_COLS = 80
DEFAULT_ROWS = 60
CELL_PX      = 10
CANVAS_W     = DEFAULT_COLS * CELL_PX
CANVAS_H     = DEFAULT_ROWS * CELL_PX
FRAME_MS     = 30


# --- Known Preset Patterns ---
PRESETS = {
    "Glider": [
        (0, 1), (1, 2), (2, 0), (2, 1), (2, 2)
    ],
    "Gosper Glider Gun": [
        (4, 0), (4, 1), (5, 0), (5, 1),
        (2, 12), (2, 13), (3, 11), (4, 10), (5, 10), (6, 10), (7, 11), (8, 12), (8, 13),
        (5, 14), (3, 15), (7, 15), (4, 16), (5, 16), (6, 16), (5, 17),
        (2, 20), (2, 21), (3, 20), (3, 21), (4, 20), (4, 21), (1, 22), (5, 22),
        (0, 24), (1, 24), (5, 24), (6, 24),
        (2, 34), (2, 35), (3, 34), (3, 35)
    ],
    "Pulsar (Oscillator)": [
        # Row offsets from center
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


def make_random_grid(rows, cols, density=0.30):
    """Generate a random grid with given alive-cell density."""
    return [[1 if random.random() < density else 0 for _ in range(cols)]
            for _ in range(rows)]


def make_empty_grid(rows, cols):
    """Generate an empty (all dead) grid."""
    return [[0] * cols for _ in range(rows)]


def fast_next_generation(current, rows, cols):
    """
    Optimized Conway's Game of Life step using zero-padded boundary.
    Eliminates inner coordinate-check branches for high throughput.
    Returns: (new_grid, changed_cells_list)
    where changed_cells_list is a list of (row, col, new_state) tuples.
    """
    # Create zero-padded 2D grid for fast branchless interior sums
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
            # Sum the 8 neighbours directly without inner loop overhead
            neighbours = (
                prev_r[c - 1] + prev_r[c] + prev_r[c + 1] +
                curr_r[c - 1]             + curr_r[c + 1] +
                next_r[c - 1] + next_r[c] + next_r[c + 1]
            )

            cell = orig_r[c - 1]
            # Standard Conway's Game of Life rule:
            # 1 survives if 2 or 3 neighbours; 0 born if 3 neighbours
            val = 1 if (neighbours == 3 or (cell == 1 and neighbours == 2)) else 0
            new_row[c - 1] = val

            if val != cell:
                changed.append((r - 1, c - 1, val))

        nxt.append(new_row)

    return nxt, changed


def run_cpu_benchmark(rows=1000, cols=1000, iterations=100):
    """
    Standalone CPU benchmark for Game of Life in Python.
    Returns execution time, mega-updates/sec, and final live cell count.
    """
    grid = make_random_grid(rows, cols, density=0.30)
    start_time = time.perf_counter()

    for _ in range(iterations):
        grid, _ = fast_next_generation(grid, rows, cols)

    elapsed = time.perf_counter() - start_time
    total_updates = rows * cols * iterations
    mega_updates = total_updates / (elapsed * 1e6)
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


class GameOfLifeApp(tk.Tk):
    """
    High-Performance Visualizer for Conway's Game of Life.
    Features incremental canvas item updating, interactive drawing,
    preset patterns, and multi-threaded simulation.
    """

    def __init__(self, rows=DEFAULT_ROWS, cols=DEFAULT_COLS):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.cell_px = CELL_PX

        self.title("HPC - Conway's Game of Life (Optimized GUI)")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)

        # State data
        self.grid_data = make_random_grid(self.rows, self.cols)
        self.generation = 0
        self.running = False
        self._sim_thread = None
        self._pending_grid = None
        self._pending_changed = None
        self._lock = threading.Lock()
        self._rect_items = [[None] * self.cols for _ in range(self.rows)]
        self._last_fps_time = time.time()
        self._gen_count_for_fps = 0
        self._current_gps = 0.0

        # Build UI layout
        self._build_header()
        self._build_body()
        self._build_footer()
        self._init_canvas_items()
        self._update_stats()
        self._refresh_loop()

        # Handle window close gracefully
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG_DARK, pady=12)
        hdr.pack(fill="x", padx=16)

        # Title branding
        title_frm = tk.Frame(hdr, bg=BG_DARK)
        title_frm.pack(side="left")

        tk.Label(title_frm, text="HPC",
                 font=("Courier New", 20, "bold"),
                 fg=ACCENT, bg=BG_DARK).pack(side="left", padx=(0, 6))

        tk.Label(title_frm, text="Conway's Game of Life",
                 font=("Segoe UI", 16, "bold"),
                 fg=TEXT_PRIMARY, bg=BG_DARK).pack(side="left")

        tk.Label(title_frm, text=" (Fast Engine)",
                 font=("Segoe UI", 10, "italic"),
                 fg=ACCENT2, bg=BG_DARK).pack(side="left", pady=(4, 0))

        # Controls panel
        ctrl = tk.Frame(hdr, bg=BG_DARK)
        ctrl.pack(side="right")

        # Iterations spinbox
        tk.Label(ctrl, text="Target Iters",
                 font=("Segoe UI", 9, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_DARK).grid(row=0, column=0, sticky="w", padx=(0, 4))

        self.iter_var = tk.IntVar(value=500)
        tk.Spinbox(ctrl, from_=1, to=100000, textvariable=self.iter_var,
                   width=6, font=("Segoe UI", 11, "bold"),
                   bg=BG_CARD, fg=ACCENT, buttonbackground=BG_PANEL,
                   insertbackground=ACCENT, relief="flat",
                   highlightthickness=1, highlightcolor=ACCENT,
                   highlightbackground=ACCENT2).grid(row=0, column=1, padx=(0, 14))

        # Speed slider
        tk.Label(ctrl, text="Speed",
                 font=("Segoe UI", 9, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_DARK).grid(row=0, column=2, sticky="w", padx=(0, 4))

        self.speed_var = tk.IntVar(value=65)
        tk.Scale(ctrl, from_=1, to=100, orient="horizontal",
                 variable=self.speed_var, length=90,
                 bg=BG_DARK, fg=ACCENT, troughcolor=BG_CARD,
                 highlightthickness=0, sliderrelief="flat",
                 showvalue=False).grid(row=0, column=3, padx=(0, 14))

        # Action buttons
        self.btn_start = self._make_btn(ctrl, "▶ Start", BTN_START, BTN_TEXT,  self._on_start, col=4)
        self.btn_stop  = self._make_btn(ctrl, "⏸ Pause", BTN_STOP,  "#ffffff", self._on_stop,  col=5, state="disabled")
        self.btn_step  = self._make_btn(ctrl, "⏭ Step",  BTN_STEP,  "#ffffff", self._on_step,  col=6)
        self.btn_reset = self._make_btn(ctrl, "🔄 Reset", BTN_RESET, "#ffffff", self._on_reset, col=7)
        self.btn_clear = self._make_btn(ctrl, "🗑 Clear", BTN_CLEAR, "#ffffff", self._on_clear, col=8)

    def _make_btn(self, parent, text, bg, fg, cmd, col, state="normal"):
        btn = tk.Button(parent, text=text,
                         font=("Segoe UI", 9, "bold"),
                         bg=bg, fg=fg,
                         activebackground=bg, activeforeground=fg,
                         relief="flat", cursor="hand2",
                         padx=10, pady=5,
                         command=cmd, state=state)
        btn.grid(row=0, column=col, padx=3)
        btn.bind("<Enter>", lambda e: btn.config(relief="groove"))
        btn.bind("<Leave>", lambda e: btn.config(relief="flat"))
        return btn

    def _build_body(self):
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(padx=16, pady=(0, 8))

        # Canvas container
        canvas_card = tk.Frame(body, bg=BG_CARD,
                                highlightbackground=ACCENT2, highlightthickness=1)
        canvas_card.pack(side="left", padx=(0, 14))

        top_bar = tk.Frame(canvas_card, bg=BG_CARD)
        top_bar.pack(fill="x", padx=6, pady=4)

        tk.Label(top_bar, text=f"GRID CANVAS ({self.cols}x{self.rows})  -  Click / Drag to Draw Cells",
                 font=("Courier New", 8, "bold"),
                 fg=ACCENT, bg=BG_CARD).pack(side="left")

        # Preset selector dropdown
        preset_frame = tk.Frame(top_bar, bg=BG_CARD)
        preset_frame.pack(side="right")
        tk.Label(preset_frame, text="Preset:", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_CARD).pack(side="left", padx=(0, 4))

        self.preset_var = tk.StringVar(value="Choose Pattern...")
        preset_options = ["Choose Pattern...", "Random (30%)"] + list(PRESETS.keys())
        preset_cb = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                 values=preset_options, state="readonly", width=18)
        preset_cb.pack(side="left")
        preset_cb.bind("<<ComboboxSelected>>", self._on_preset_selected)

        # Simulation Canvas
        self.canvas = tk.Canvas(canvas_card,
                                width=CANVAS_W, height=CANVAS_H,
                                bg=DEAD_COLOR, highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(padx=2, pady=(0, 2))

        # Mouse drawing event bindings
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag_draw)
        self.canvas.bind("<Button-3>", self._on_canvas_right_click)
        self.canvas.bind("<B3-Motion>", self._on_canvas_drag_erase)

        # Right-side stats & info panel
        side = tk.Frame(body, bg=BG_DARK)
        side.pack(side="left", fill="y")

        self._stat_cards = {}
        cards_info = [
            ("generation", "Generation", "0",     ACCENT),
            ("live_cells", "Live Cells", "0",     "#10b981"),
            ("gps",        "Speed (Gen/s)", "0.0", "#38bdf8"),
            ("progress",   "Progress",   "0.0 %", "#f59e0b"),
            ("iterations", "Target Iters","500",  ACCENT2),
        ]
        for key, label, init, colour in cards_info:
            self._stat_cards[key] = self._make_stat_card(side, label, init, colour)

        # Legend card
        leg = tk.Frame(side, bg=BG_CARD, padx=10, pady=8,
                       highlightbackground=ACCENT2, highlightthickness=1)
        leg.pack(fill="x", pady=(10, 0))
        tk.Label(leg, text="CANVAS CONTROLS", font=("Courier New", 8, "bold"),
                 fg=ACCENT, bg=BG_CARD).pack(anchor="w")
        for desc in [
            "• Left Click / Drag: Paint alive",
            "• Right Click / Drag: Erase cells",
            "• Presets: Stamp classic patterns",
            "• Step: Single generation tick",
        ]:
            tk.Label(leg, text=desc, fg=TEXT_SECONDARY,
                     bg=BG_CARD, font=("Segoe UI", 8)).pack(anchor="w", pady=1)

        # Conway rules card
        rules = tk.Frame(side, bg=BG_CARD, padx=10, pady=8,
                         highlightbackground=ACCENT2, highlightthickness=1)
        rules.pack(fill="x", pady=(8, 0))
        tk.Label(rules, text="CONWAY'S RULES",
                 font=("Courier New", 8, "bold"),
                 fg=ACCENT2, bg=BG_CARD).pack(anchor="w", pady=(0, 2))
        for line in [
            "1. Live + 2-3 nbrs -> Survives",
            "2. Live + <2 nbrs  -> Under-pop",
            "3. Live + >3 nbrs  -> Over-pop",
            "4. Dead + 3 nbrs   -> Reproduction",
        ]:
            tk.Label(rules, text=line, fg=TEXT_SECONDARY, bg=BG_CARD,
                     font=("Segoe UI", 8), anchor="w").pack(anchor="w")

    def _make_stat_card(self, parent, label, init_val, colour):
        card = tk.Frame(parent, bg=BG_CARD, padx=12, pady=6,
                        highlightbackground=colour, highlightthickness=1)
        card.pack(fill="x", pady=3)
        tk.Label(card, text=label, font=("Segoe UI", 8, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_CARD).pack(anchor="w")
        val_lbl = tk.Label(card, text=init_val,
                           font=("Courier New", 16, "bold"),
                           fg=colour, bg=BG_CARD)
        val_lbl.pack(anchor="w")
        return val_lbl

    def _build_footer(self):
        foot = tk.Frame(self, bg=BG_PANEL, pady=6)
        foot.pack(fill="x")

        self.progress_var = tk.DoubleVar(value=0.0)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("GOL.Horizontal.TProgressbar",
                        troughcolor=BG_CARD, background=ACCENT,
                        bordercolor=BG_PANEL,
                        lightcolor=ACCENT, darkcolor=ACCENT)
        ttk.Progressbar(foot, style="GOL.Horizontal.TProgressbar",
                        variable=self.progress_var,
                        maximum=100, length=CANVAS_W + 200).pack(padx=16, pady=2)

        self.status_lbl = tk.Label(foot,
                                   text="Ready  -  Click canvas to draw cells or press Start",
                                   font=("Segoe UI", 9),
                                   fg=TEXT_SECONDARY, bg=BG_PANEL)
        self.status_lbl.pack()

    def _init_canvas_items(self):
        """
        Pre-create all rectangle items ONCE on initialization.
        Drastically reduces garbage collection and Tkinter calls during simulation.
        """
        self.canvas.delete("all")
        for r in range(self.rows):
            y0 = r * self.cell_px
            y1 = y0 + self.cell_px - 1
            for c in range(self.cols):
                x0 = c * self.cell_px
                x1 = x0 + self.cell_px - 1
                color = ALIVE_COLOR if self.grid_data[r][c] else DEAD_COLOR
                rect_id = self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    fill=color, outline=GRID_COLOR, width=1
                )
                self._rect_items[r][c] = rect_id

    def _update_canvas_incremental(self, changed_cells):
        """Update only the specific cells that flipped state."""
        for r, c, val in changed_cells:
            color = ALIVE_COLOR if val else DEAD_COLOR
            self.canvas.itemconfig(self._rect_items[r][c], fill=color)

    def _update_canvas_full(self):
        """Full canvas state synchronizer."""
        for r in range(self.rows):
            for c in range(self.cols):
                color = ALIVE_COLOR if self.grid_data[r][c] else DEAD_COLOR
                self.canvas.itemconfig(self._rect_items[r][c], fill=color)

    def _refresh_loop(self):
        """Main Tkinter refresh loop consuming pending updates from the worker thread."""
        with self._lock:
            if self._pending_grid is not None:
                self.grid_data = self._pending_grid
                changed = self._pending_changed
                self._pending_grid = None
                self._pending_changed = None

                if changed is not None:
                    self._update_canvas_incremental(changed)
                else:
                    self._update_canvas_full()

                self._gen_count_for_fps += 1
                now = time.time()
                dt = now - self._last_fps_time
                if dt >= 0.5:
                    self._current_gps = self._gen_count_for_fps / dt
                    self._last_fps_time = now
                    self._gen_count_for_fps = 0

                self._update_stats()

        self.after(FRAME_MS, self._refresh_loop)

    def _update_stats(self):
        live = sum(sum(row) for row in self.grid_data)
        max_iter = self.iter_var.get()
        pct = (self.generation / max_iter * 100) if max_iter > 0 else 0

        self._stat_cards["generation"].config(text=f"{self.generation:,}")
        self._stat_cards["live_cells"].config(text=f"{live:,}")
        self._stat_cards["gps"].config(text=f"{self._current_gps:.1f}")
        self._stat_cards["progress"].config(text=f"{pct:.1f} %")
        self._stat_cards["iterations"].config(text=f"{max_iter:,}")
        self.progress_var.set(min(pct, 100.0))

    def _simulate(self, total_iter):
        """Worker thread executing optimized simulation steps."""
        current = [row[:] for row in self.grid_data]

        while self.running:
            if self.generation >= total_iter:
                break

            current, changed = fast_next_generation(current, self.rows, self.cols)
            self.generation += 1

            with self._lock:
                self._pending_grid = current
                self._pending_changed = changed

            # Map speed slider (1..100) to delay (200ms down to 1ms)
            speed = self.speed_var.get()
            delay = max(0.001, 0.200 - (speed - 1) * (0.199 / 99))
            time.sleep(delay)

        self.running = False
        self.after(0, self._on_sim_done)

    def _on_sim_done(self):
        self._set_running_state(False)
        self._current_gps = 0.0
        self._update_stats()
        if self.generation >= self.iter_var.get():
            self.status_lbl.config(
                text=f"Simulation complete  -  Reached {self.generation:,} generations",
                fg="#10b981"
            )
        else:
            self.status_lbl.config(
                text=f"Paused at generation {self.generation:,}",
                fg=BTN_STOP
            )

    def _on_start(self):
        if self.running:
            return
        try:
            total = int(self.iter_var.get())
            if total < 1:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid Target", "Please enter a valid target iteration (>= 1).")
            return

        self.running = True
        self._set_running_state(True)
        self.status_lbl.config(
            text=f"Running simulation  -  Target: {total:,} generations",
            fg=ACCENT
        )
        self._last_fps_time = time.time()
        self._gen_count_for_fps = 0
        self._sim_thread = threading.Thread(
            target=self._simulate, args=(total,), daemon=True
        )
        self._sim_thread.start()

    def _on_stop(self):
        self.running = False
        self._current_gps = 0.0
        self.status_lbl.config(text="Paused  -  Press Start or Step to continue", fg="#f59e0b")
        self._set_running_state(False)

    def _on_step(self):
        """Advance exactly 1 generation step while paused."""
        if self.running:
            return
        nxt, changed = fast_next_generation(self.grid_data, self.rows, self.cols)
        self.grid_data = nxt
        self.generation += 1
        self._update_canvas_incremental(changed)
        self._current_gps = 0.0
        self._update_stats()
        self.status_lbl.config(
            text=f"Stepped to generation {self.generation:,}",
            fg=BTN_STEP
        )

    def _on_reset(self):
        self.running = False
        self.generation = 0
        self._current_gps = 0.0
        self.grid_data = make_random_grid(self.rows, self.cols)
        with self._lock:
            self._pending_grid = None
            self._pending_changed = None
        self._update_canvas_full()
        self._update_stats()
        self._set_running_state(False)
        self.status_lbl.config(
            text="Grid randomized (30% density)  -  Ready to start",
            fg=TEXT_SECONDARY
        )

    def _on_clear(self):
        self.running = False
        self.generation = 0
        self._current_gps = 0.0
        self.grid_data = make_empty_grid(self.rows, self.cols)
        with self._lock:
            self._pending_grid = None
            self._pending_changed = None
        self._update_canvas_full()
        self._update_stats()
        self._set_running_state(False)
        self.status_lbl.config(
            text="Grid cleared  -  Click canvas to draw cells or load a preset",
            fg=TEXT_SECONDARY
        )

    def _on_preset_selected(self, event=None):
        name = self.preset_var.get()
        if name == "Choose Pattern...":
            return

        self.running = False
        self._set_running_state(False)
        self.generation = 0
        self._current_gps = 0.0

        if name == "Random (30%)":
            self.grid_data = make_random_grid(self.rows, self.cols)
        elif name in PRESETS:
            coords = PRESETS[name]
            self.grid_data = make_empty_grid(self.rows, self.cols)
            center_r = self.rows // 2
            center_c = self.cols // 2

            for dr, dc in coords:
                r = center_r + dr
                c = center_c + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    self.grid_data[r][c] = 1

        self._update_canvas_full()
        self._update_stats()
        self.status_lbl.config(text=f"Loaded preset pattern: {name}", fg=ACCENT)

    def _on_canvas_click(self, event):
        """Toggle cell alive/dead on left click."""
        c = event.x // self.cell_px
        r = event.y // self.cell_px
        if 0 <= r < self.rows and 0 <= c < self.cols:
            new_val = 0 if self.grid_data[r][c] else 1
            self.grid_data[r][c] = new_val
            color = ALIVE_COLOR if new_val else DEAD_COLOR
            self.canvas.itemconfig(self._rect_items[r][c], fill=color)
            self._update_stats()

    def _on_canvas_drag_draw(self, event):
        """Paint alive cells on drag."""
        c = event.x // self.cell_px
        r = event.y // self.cell_px
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.grid_data[r][c] != 1:
                self.grid_data[r][c] = 1
                self.canvas.itemconfig(self._rect_items[r][c], fill=ALIVE_COLOR)
                self._update_stats()

    def _on_canvas_right_click(self, event):
        """Erase cell on right click."""
        c = event.x // self.cell_px
        r = event.y // self.cell_px
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid_data[r][c] = 0
            self.canvas.itemconfig(self._rect_items[r][c], fill=DEAD_COLOR)
            self._update_stats()

    def _on_canvas_drag_erase(self, event):
        """Erase cells on right-drag."""
        c = event.x // self.cell_px
        r = event.y // self.cell_px
        if 0 <= r < self.rows and 0 <= c < self.cols:
            if self.grid_data[r][c] != 0:
                self.grid_data[r][c] = 0
                self.canvas.itemconfig(self._rect_items[r][c], fill=DEAD_COLOR)
                self._update_stats()

    def _set_running_state(self, is_running):
        self.btn_start.config(state="disabled" if is_running else "normal")
        self.btn_stop.config(state="normal"    if is_running else "disabled")
        self.btn_step.config(state="disabled" if is_running else "normal")

    def _on_close(self):
        """Gracefully terminate worker thread on application exit."""
        self.running = False
        self.destroy()


def launch_gui(rows=DEFAULT_ROWS, cols=DEFAULT_COLS):
    """Convenience launcher for the GUI."""
    app = GameOfLifeApp(rows=rows, cols=cols)
    app.mainloop()


if __name__ == "__main__":
    launch_gui()