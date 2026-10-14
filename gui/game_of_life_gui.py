"""
HPC Game of Life - GUI Visualizer
Conway's Game of Life with CPU simulation and dynamic canvas animation.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import random
import time
import threading

# --- Colour palette ---
BG_DARK        = "#0d0f1a"
BG_PANEL       = "#111827"
BG_CARD        = "#1a2035"
ACCENT         = "#00d4ff"
ACCENT2        = "#7c3aed"
ALIVE_COLOR    = "#00d4ff"
DEAD_COLOR     = "#0d1526"
GRID_COLOR     = "#1a2a3a"
TEXT_PRIMARY   = "#e2e8f0"
TEXT_SECONDARY = "#64748b"
BTN_START      = "#00d4ff"
BTN_STOP       = "#ef4444"
BTN_RESET      = "#7c3aed"
BTN_TEXT       = "#0d0f1a"

# --- Simulation parameters ---
GRID_COLS  = 80
GRID_ROWS  = 60
CELL_PX    = 9
CANVAS_W   = GRID_COLS * CELL_PX
CANVAS_H   = GRID_ROWS * CELL_PX
FRAME_MS   = 60


def make_random_grid(rows, cols):
    return [[1 if random.random() < 0.30 else 0 for _ in range(cols)]
            for _ in range(rows)]


def next_generation(current, rows, cols):
    nxt = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            neighbours = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbours += current[nr][nc]
            if current[r][c] == 1:
                nxt[r][c] = 1 if neighbours in (2, 3) else 0
            else:
                nxt[r][c] = 1 if neighbours == 3 else 0
    return nxt


class GameOfLifeApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("HPC - Conway's Game of Life")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)

        self.grid_data = make_random_grid(GRID_ROWS, GRID_COLS)
        self.generation  = 0
        self.running     = False
        self._sim_thread = None
        self._pending    = None
        self._lock       = threading.Lock()

        self._build_header()
        self._build_body()
        self._build_footer()
        self._draw_canvas()
        self._update_stats()
        self._refresh_loop()

    def _build_header(self):
        hdr = tk.Frame(self, bg=BG_DARK, pady=16)
        hdr.pack(fill="x", padx=20)

        title_frm = tk.Frame(hdr, bg=BG_DARK)
        title_frm.pack(side="left")

        tk.Label(title_frm, text="HPC",
                 font=("Courier New", 22, "bold"),
                 fg=ACCENT, bg=BG_DARK).pack(side="left", padx=(0, 8))

        tk.Label(title_frm, text="Conway's Game of Life",
                 font=("Segoe UI", 18, "bold"),
                 fg=TEXT_PRIMARY, bg=BG_DARK).pack(side="left")

        tk.Label(title_frm, text="  - CPU Simulation",
                 font=("Segoe UI", 11),
                 fg=TEXT_SECONDARY, bg=BG_DARK).pack(side="left", pady=(6, 0))

        ctrl = tk.Frame(hdr, bg=BG_DARK)
        ctrl.pack(side="right")

        tk.Label(ctrl, text="Iterations",
                 font=("Segoe UI", 10, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_DARK).grid(row=0, column=0, sticky="w", padx=(0, 6))

        self.iter_var = tk.IntVar(value=100)
        tk.Spinbox(ctrl, from_=1, to=10000, textvariable=self.iter_var,
                   width=7, font=("Segoe UI", 12, "bold"),
                   bg=BG_CARD, fg=ACCENT, buttonbackground=BG_PANEL,
                   insertbackground=ACCENT, relief="flat",
                   highlightthickness=1, highlightcolor=ACCENT,
                   highlightbackground=ACCENT2).grid(row=0, column=1, padx=(0, 20))

        tk.Label(ctrl, text="Speed",
                 font=("Segoe UI", 10, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_DARK).grid(row=0, column=2, sticky="w", padx=(0, 6))

        self.speed_var = tk.IntVar(value=50)
        tk.Scale(ctrl, from_=1, to=100, orient="horizontal",
                 variable=self.speed_var, length=110,
                 bg=BG_DARK, fg=ACCENT, troughcolor=BG_CARD,
                 highlightthickness=0, sliderrelief="flat",
                 showvalue=False).grid(row=0, column=3, padx=(0, 20))

        self.btn_start = self._make_btn(ctrl, "  Start",  BTN_START, BTN_TEXT,  self._on_start, col=4)
        self.btn_stop  = self._make_btn(ctrl, "  Stop",   BTN_STOP,  "#ffffff", self._on_stop,  col=5, state="disabled")
        self.btn_reset = self._make_btn(ctrl, "  Reset",  BTN_RESET, "#ffffff", self._on_reset, col=6)

    def _make_btn(self, parent, text, bg, fg, cmd, col, state="normal"):
        btn = tk.Button(parent, text=text,
                        font=("Segoe UI", 10, "bold"),
                        bg=bg, fg=fg,
                        activebackground=bg, activeforeground=fg,
                        relief="flat", cursor="hand2",
                        padx=14, pady=7,
                        command=cmd, state=state)
        btn.grid(row=0, column=col, padx=4)
        btn.bind("<Enter>", lambda e: btn.config(relief="groove"))
        btn.bind("<Leave>", lambda e: btn.config(relief="flat"))
        return btn

    def _build_body(self):
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(padx=20, pady=(0, 10))

        canvas_card = tk.Frame(body, bg=BG_CARD,
                               highlightbackground=ACCENT2, highlightthickness=1)
        canvas_card.pack(side="left", padx=(0, 16))

        tk.Label(canvas_card, text="SIMULATION CANVAS",
                 font=("Courier New", 9, "bold"),
                 fg=ACCENT2, bg=BG_CARD, pady=6).pack()

        self.canvas = tk.Canvas(canvas_card,
                                width=CANVAS_W, height=CANVAS_H,
                                bg=BG_DARK, highlightthickness=0)
        self.canvas.pack(padx=2, pady=(0, 2))

        side = tk.Frame(body, bg=BG_DARK)
        side.pack(side="left", fill="y")

        self._stat_cards = {}
        for key, label, init, colour in [
            ("generation", "Generation", "0",    ACCENT),
            ("live_cells", "Live Cells", "0",    "#10b981"),
            ("progress",   "Progress",   "0 %",  "#f59e0b"),
            ("iterations", "Max Iters",  "100",  ACCENT2),
        ]:
            self._stat_cards[key] = self._make_stat_card(side, label, init, colour)

        leg = tk.Frame(side, bg=BG_CARD, padx=12, pady=10,
                       highlightbackground=ACCENT2, highlightthickness=1)
        leg.pack(fill="x", pady=(16, 0))
        tk.Label(leg, text="LEGEND", font=("Courier New", 8, "bold"),
                 fg=ACCENT2, bg=BG_CARD).pack(anchor="w")
        for col, lbl in [(ALIVE_COLOR, "Alive cell"),
                         ("#1e3a5f",   "Dead cell"),
                         (ACCENT2,     "Grid line")]:
            rf = tk.Frame(leg, bg=BG_CARD); rf.pack(anchor="w", pady=2)
            tk.Label(rf, text="  ", bg=col, fg=col,
                     font=("Segoe UI", 10)).pack(side="left", padx=(0,4))
            tk.Label(rf, text=lbl, fg=TEXT_SECONDARY,
                     bg=BG_CARD, font=("Segoe UI", 9)).pack(side="left")

        rules = tk.Frame(side, bg=BG_CARD, padx=12, pady=10,
                         highlightbackground=ACCENT2, highlightthickness=1)
        rules.pack(fill="x", pady=(12, 0))
        tk.Label(rules, text="CONWAY'S RULES",
                 font=("Courier New", 8, "bold"),
                 fg=ACCENT2, bg=BG_CARD).pack(anchor="w", pady=(0, 4))
        for line in [
            "Live + 2-3 nbrs -> Survive",
            "Live + <2 nbrs  -> Die (under-pop)",
            "Live + >3 nbrs  -> Die (over-pop)",
            "Dead + 3 nbrs   -> Born",
        ]:
            tk.Label(rules, text="  " + line, fg=TEXT_SECONDARY, bg=BG_CARD,
                     font=("Segoe UI", 9), anchor="w",
                     justify="left").pack(anchor="w")

    def _make_stat_card(self, parent, label, init_val, colour):
        card = tk.Frame(parent, bg=BG_CARD, padx=16, pady=12,
                        highlightbackground=colour, highlightthickness=1)
        card.pack(fill="x", pady=5)
        tk.Label(card, text=label, font=("Segoe UI", 9, "bold"),
                 fg=TEXT_SECONDARY, bg=BG_CARD).pack(anchor="w")
        val_lbl = tk.Label(card, text=init_val,
                           font=("Courier New", 20, "bold"),
                           fg=colour, bg=BG_CARD)
        val_lbl.pack(anchor="w")
        return val_lbl

    def _build_footer(self):
        foot = tk.Frame(self, bg=BG_PANEL, pady=8)
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
                        maximum=100, length=CANVAS_W + 240).pack(padx=20, pady=4)

        self.status_lbl = tk.Label(foot,
                                   text="Ready  -  Adjust iterations and press Start",
                                   font=("Segoe UI", 9),
                                   fg=TEXT_SECONDARY, bg=BG_PANEL)
        self.status_lbl.pack()

    def _draw_canvas(self):
        self.canvas.delete("all")
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x0 = c * CELL_PX; y0 = r * CELL_PX
                x1 = x0 + CELL_PX - 1; y1 = y0 + CELL_PX - 1
                color = ALIVE_COLOR if self.grid_data[r][c] else DEAD_COLOR
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill=color, outline=GRID_COLOR)

    def _refresh_loop(self):
        with self._lock:
            if self._pending is not None:
                self.grid_data = self._pending
                self._pending  = None
                self._draw_canvas()
                self._update_stats()
        self.after(FRAME_MS, self._refresh_loop)

    def _update_stats(self):
        live     = sum(sum(row) for row in self.grid_data)
        max_iter = self.iter_var.get()
        pct      = (self.generation / max_iter * 100) if max_iter else 0
        self._stat_cards["generation"].config(text=f"{self.generation:,}")
        self._stat_cards["live_cells"].config(text=f"{live:,}")
        self._stat_cards["progress"].config(text=f"{pct:.1f} %")
        self._stat_cards["iterations"].config(text=f"{max_iter:,}")
        self.progress_var.set(min(pct, 100))

    def _simulate(self, total_iter):
        current = [row[:] for row in self.grid_data]
        for _ in range(1, total_iter + 1):
            if not self.running:
                break
            current = next_generation(current, GRID_ROWS, GRID_COLS)
            self.generation += 1
            with self._lock:
                self._pending = [row[:] for row in current]
            speed = self.speed_var.get()
            delay = max(0.005, 0.200 - (speed - 1) * (0.195 / 99))
            time.sleep(delay)
        self.running = False
        self.after(0, self._on_sim_done)

    def _on_sim_done(self):
        self._set_running_state(False)
        if self.generation >= self.iter_var.get():
            self.status_lbl.config(
                text=f"Simulation complete  -  {self.generation:,} generations", fg="#10b981")
        else:
            self.status_lbl.config(
                text=f"Stopped at generation {self.generation:,}", fg=BTN_STOP)

    def _on_start(self):
        if self.running:
            return
        try:
            total = int(self.iter_var.get())
            if total < 1:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid input", "Please enter a valid number of iterations (>= 1).")
            return
        self.running = True
        self._set_running_state(True)
        self.status_lbl.config(
            text=f"Running simulation  -  target {total:,} generations", fg=ACCENT)
        self._sim_thread = threading.Thread(
            target=self._simulate, args=(total,), daemon=True)
        self._sim_thread.start()

    def _on_stop(self):
        self.running = False
        self.status_lbl.config(text="Paused - press Start to continue", fg="#f59e0b")
        self._set_running_state(False)

    def _on_reset(self):
        self.running    = False
        self.generation = 0
        self.grid_data  = make_random_grid(GRID_ROWS, GRID_COLS)
        with self._lock:
            self._pending = None
        self._draw_canvas()
        self._update_stats()
        self._set_running_state(False)
        self.status_lbl.config(
            text="Grid reset  -  Adjust iterations and press Start", fg=TEXT_SECONDARY)

    def _set_running_state(self, is_running):
        self.btn_start.config(state="disabled" if is_running else "normal")
        self.btn_stop.config(state="normal"    if is_running else "disabled")


if __name__ == "__main__":
    app = GameOfLifeApp()
    app.mainloop()