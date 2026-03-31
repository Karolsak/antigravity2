"""
3-Phase Induction Motor Analysis - Complete Engineering Suite
=============================================================
Comprehensive tkinter application for analysis of a 3-phase induction motor.

Problem parameters (default):
  VL = 400 V, 50 Hz, 4-pole, star-connected
  R2 = 0.1 Ω/phase (referred to stator)
  X2 = 1.0 Ω/phase (referred to stator)
  N1/N2 = 4
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time


# ---------------------------------------------------------------------------
# Main Application Class
# ---------------------------------------------------------------------------

class InductionMotorApp:
    """Main application class containing all 10 analysis tabs."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(
            "3-Phase Induction Motor Analysis - Complete Engineering Suite"
        )
        self.root.geometry("1400x900")
        self.root.resizable(True, True)

        # ── Motor parameters (tk variables) ──────────────────────────────
        self.VL            = tk.DoubleVar(value=400.0)
        self.R2            = tk.DoubleVar(value=0.1)
        self.X2            = tk.DoubleVar(value=1.0)
        self.turns_ratio   = tk.DoubleVar(value=4.0)
        self.frequency     = tk.DoubleVar(value=50.0)
        self.poles         = tk.IntVar(value=4)
        self.slip_fl_pct   = tk.DoubleVar(value=4.0)   # full-load slip in %

        # ── Fault / simulation flags ──────────────────────────────────────
        self.sim_running   = False
        self._sim_thread   = None

        # ── Layout ───────────────────────────────────────────────────────
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # ── Build tabs ───────────────────────────────────────────────────
        self.create_tab_1_overview()
        self.create_tab_2_input_parameters()
        self.create_tab_3_torque_speed()
        self.create_tab_4_fault_current()
        self.create_tab_5_protection()
        self.create_tab_6_speed_controller()
        self.create_tab_7_thermal()
        self.create_tab_8_economic()
        self.create_tab_9_harmonic()
        self.create_tab_10_comprehensive()

    # ======================================================================
    # Core calculation
    # ======================================================================

    def calculate_motor_params(self) -> dict:
        """Return dict of all derived motor quantities."""
        VL      = self.VL.get()
        R2      = max(self.R2.get(), 1e-6)
        X2      = max(self.X2.get(), 1e-6)
        n_ratio = max(self.turns_ratio.get(), 1e-3)
        f       = max(self.frequency.get(), 1.0)
        poles   = max(int(self.poles.get()), 2)
        s_fl    = np.clip(self.slip_fl_pct.get() / 100.0, 0.001, 0.999)

        V1 = VL / np.sqrt(3)
        Ns = 120.0 * f / poles          # synchronous speed, rpm
        ws = 2.0 * np.pi * Ns / 60.0   # synchronous angular speed, rad/s

        # Torque formula helper
        def torque(s_val):
            r = R2 / np.clip(s_val, 1e-6, None)
            return (3.0 / ws) * V1**2 * r / (r**2 + X2**2)

        T_st  = torque(1.0)
        T_fl  = torque(s_fl)
        s_m   = R2 / X2
        T_max = (3.0 / ws) * V1**2 / (2.0 * X2)
        N_maxT = Ns * (1.0 - s_m)

        ratio_st = np.clip(T_st / T_max, 0, None) if T_max > 0 else 0.0
        ratio_fl = np.clip(T_fl / T_max, 0, None) if T_max > 0 else 0.0
        R_add    = (X2 - R2) / n_ratio**2

        return {
            "VL": VL, "V1": V1, "R2": R2, "X2": X2,
            "f": f, "poles": poles, "Ns": Ns, "ws": ws,
            "s_fl": s_fl, "n_ratio": n_ratio,
            "T_st": T_st, "T_fl": T_fl, "T_max": T_max,
            "s_m": s_m, "N_maxT": N_maxT,
            "ratio_st": ratio_st, "ratio_fl": ratio_fl,
            "R_add": R_add,
        }

    # ======================================================================
    # Helper: embed Figure in a frame
    # ======================================================================

    def _embed_figure(self, fig: Figure, parent: tk.Widget,
                      row: int = 0, col: int = 0) -> FigureCanvasTkAgg:
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().grid(row=row, column=col, sticky="nsew")
        return canvas

    # ======================================================================
    # TAB 1 – Overview & Theory
    # ======================================================================

    def create_tab_1_overview(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Overview & Theory")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        txt_frame = ttk.Frame(frame)
        txt_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        txt_frame.grid_rowconfigure(0, weight=1)
        txt_frame.grid_columnconfigure(0, weight=1)

        txt = tk.Text(txt_frame, wrap=tk.WORD, font=("Courier", 10),
                      background="#fafafa")
        vsb = ttk.Scrollbar(txt_frame, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        txt.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # Colour tags
        txt.tag_configure("title",     foreground="#1a237e",
                          font=("Arial", 13, "bold"))
        txt.tag_configure("header",    foreground="#0d47a1",
                          font=("Arial", 11, "bold"))
        txt.tag_configure("formula",   foreground="#006064",
                          font=("Courier", 10))
        txt.tag_configure("result",    foreground="#1b5e20",
                          font=("Arial", 10, "bold"))
        txt.tag_configure("highlight", foreground="#b71c1c",
                          font=("Arial", 10, "bold"))
        txt.tag_configure("normal",    foreground="#212121",
                          font=("Arial", 10))

        p = self.calculate_motor_params()

        content = [
            ("=" * 78 + "\n",                                         "title"),
            ("  3-PHASE INDUCTION MOTOR ANALYSIS  –  COMPLETE PROBLEM SOLUTION\n",
             "title"),
            ("=" * 78 + "\n\n",                                        "title"),

            ("PROBLEM STATEMENT\n",                                   "header"),
            ("-" * 50 + "\n",                                         "normal"),
            ("A 3-phase induction motor has the following specifications:\n",
             "normal"),
            ("  • Line voltage VL          = 400 V, 50 Hz supply\n",  "normal"),
            ("  • Configuration            = 4-pole, star-connected stator\n",
             "normal"),
            ("  • Rotor resistance R2      = 0.1 Ω/phase (referred to stator)\n",
             "normal"),
            ("  • Standstill reactance X2  = 1.0 Ω/phase (referred to stator)\n",
             "normal"),
            ("  • Turns ratio N1/N2        = 4\n\n",                  "normal"),

            ("GIVEN / DERIVED PARAMETERS\n",                          "header"),
            ("-" * 50 + "\n",                                         "normal"),
            (f"  Phase voltage  V1 = VL/√3 = {p['VL']:.2f}/√3 "
             f"= {p['V1']:.4f} V\n",                                  "formula"),
            (f"  Synchronous speed  Ns = 120f/P = 120×{p['f']:.0f}/{p['poles']}"
             f" = {p['Ns']:.0f} rpm\n",                               "formula"),
            (f"  Angular speed  ωs = 2π·Ns/60 = {p['ws']:.6f} rad/s\n\n",
             "formula"),

            ("GENERAL TORQUE FORMULA\n",                              "header"),
            ("-" * 50 + "\n",                                         "normal"),
            ("  T = (3/ωs) · V1² · (R2/s) / [(R2/s)² + X2²]  (N·m)\n\n",
             "formula"),

            ("(i)  STARTING TORQUE  (s = 1)\n",                      "header"),
            ("-" * 50 + "\n",                                         "normal"),
            ("  T_st = (3/ωs) · V1² · R2 / (R2² + X2²)\n",          "formula"),
            (f"       = (3/{p['ws']:.4f}) · ({p['V1']:.4f})² · {p['R2']}"
             f" / ({p['R2']}² + {p['X2']}²)\n",                      "formula"),
            (f"       = (3/{p['ws']:.4f}) · {p['V1']**2:.4f} · {p['R2']}"
             f" / {p['R2']**2 + p['X2']**2:.4f}\n",                  "formula"),
            (f"  T_st = {p['T_st']:.4f} N·m\n\n",                    "result"),

            ("(ii) FULL-LOAD TORQUE  (s = 0.04)\n",                  "header"),
            ("-" * 50 + "\n",                                         "normal"),
            ("  T_fl = (3/ωs) · V1² · (R2/s) / [(R2/s)² + X2²]\n", "formula"),
            (f"  At s = {p['s_fl']:.4f}:  R2/s = {p['R2']}/{p['s_fl']:.4f}"
             f" = {p['R2']/p['s_fl']:.4f} Ω\n",                      "formula"),
            (f"  T_fl = {p['T_fl']:.4f} N·m\n\n",                    "result"),

            ("(iii) SLIP AT MAXIMUM TORQUE\n",                        "header"),
            ("-" * 50 + "\n",                                         "normal"),
            ("  s_m = R2 / X2\n",                                     "formula"),
            (f"      = {p['R2']} / {p['X2']} = {p['s_m']:.4f}"
             f"  ({p['s_m']*100:.1f} %)\n",                           "formula"),
            (f"  Rotor speed at T_max = Ns·(1−s_m)"
             f" = {p['Ns']:.0f}·(1−{p['s_m']:.2f})"
             f" = {p['N_maxT']:.0f} rpm\n\n",                         "result"),

            ("(iv) MAXIMUM TORQUE\n",                                 "header"),
            ("-" * 50 + "\n",                                         "normal"),
            ("  T_max = (3/ωs) · V1² / (2·X2)\n",                   "formula"),
            (f"        = (3/{p['ws']:.4f}) · {p['V1']**2:.4f}"
             f" / (2·{p['X2']})\n",                                   "formula"),
            (f"  T_max = {p['T_max']:.4f} N·m\n\n",                  "result"),

            ("(v)  TORQUE RATIOS\n",                                  "header"),
            ("-" * 50 + "\n",                                         "normal"),
            (f"  T_st / T_max = {p['T_st']:.4f} / {p['T_max']:.4f}"
             f" = {p['ratio_st']:.4f}\n",                             "formula"),
            (f"  T_fl / T_max = {p['T_fl']:.4f} / {p['T_max']:.4f}"
             f" = {p['ratio_fl']:.4f}\n\n",                           "result"),

            ("(vi) ADDITIONAL ROTOR RESISTANCE FOR T_max AT START\n", "header"),
            ("-" * 50 + "\n",                                         "normal"),
            ("  Condition: R2_total = X2  ⟹  need s_m = 1\n",       "formula"),
            (f"  Additional R (referred to stator) = X2 − R2"
             f" = {p['X2']} − {p['R2']} = {p['X2']-p['R2']:.4f} Ω\n",
             "formula"),
            (f"  Actual rotor side: R_add = (X2−R2)/(N1/N2)²"
             f" = {p['X2']-p['R2']:.4f}/{p['n_ratio']**2:.0f}\n",    "formula"),
            (f"  R_add = {p['R_add']:.6f} Ω"
             f"  = {p['R_add']*1000:.4f} mΩ\n\n",                    "result"),

            ("SUMMARY\n",                                             "header"),
            ("=" * 50 + "\n",                                         "normal"),
            (f"  Starting Torque   T_st   = {p['T_st']:.4f} N·m\n",  "highlight"),
            (f"  Full-Load Torque  T_fl   = {p['T_fl']:.4f} N·m\n",  "highlight"),
            (f"  Maximum Torque    T_max  = {p['T_max']:.4f} N·m\n", "highlight"),
            (f"  Slip at T_max     s_m    = {p['s_m']:.4f}"
             f"  ({p['s_m']*100:.1f} %)\n",                          "highlight"),
            (f"  Speed at T_max    N_maxT = {p['N_maxT']:.0f} rpm\n","highlight"),
            (f"  T_st/T_max               = {p['ratio_st']:.4f}\n",  "highlight"),
            (f"  T_fl/T_max               = {p['ratio_fl']:.4f}\n",  "highlight"),
            (f"  R_add (rotor)            = {p['R_add']:.6f} Ω\n",   "highlight"),
        ]

        for text, tag in content:
            txt.insert(tk.END, text, tag)

        txt.config(state=tk.DISABLED)

    # ======================================================================
    # TAB 2 – Input Parameters
    # ======================================================================

    def create_tab_2_input_parameters(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Input Parameters")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # ── Left: sliders ────────────────────────────────────────────────
        left = ttk.LabelFrame(frame, text="Motor Parameters", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        left.grid_columnconfigure(1, weight=1)

        slider_defs = [
            ("Line Voltage VL (V)",        self.VL,          100,  600),
            ("Rotor Resistance R2 (Ω)",    self.R2,          0.01, 2.0),
            ("Standstill Reactance X2 (Ω)", self.X2,         0.1,  5.0),
            ("Turns Ratio N1/N2",          self.turns_ratio,  1,   10),
            ("Frequency (Hz)",             self.frequency,    25,  100),
            ("Full-Load Slip (%)",         self.slip_fl_pct,  1,   15),
        ]

        for row_idx, (label, var, lo, hi) in enumerate(slider_defs):
            ttk.Label(left, text=label).grid(
                row=row_idx * 2, column=0, sticky="w", padx=4, pady=2)
            ttk.Entry(left, textvariable=var, width=8).grid(
                row=row_idx * 2, column=1, sticky="e", padx=4)
            ttk.Scale(
                left, variable=var, from_=lo, to=hi,
                orient=tk.HORIZONTAL, length=220,
                command=lambda _v: self._schedule_update_results()
            ).grid(row=row_idx * 2 + 1, column=0, columnspan=2,
                   sticky="ew", padx=4, pady=2)
            var.trace_add("write",
                          lambda *_a: self._schedule_update_results())

        # Poles radio buttons
        r = len(slider_defs)
        ttk.Label(left, text="Number of Poles").grid(
            row=r * 2, column=0, sticky="w", padx=4, pady=4)
        pf = ttk.Frame(left)
        pf.grid(row=r * 2, column=1, sticky="w")
        for p_val in (2, 4, 6, 8):
            ttk.Radiobutton(
                pf, text=str(p_val), variable=self.poles, value=p_val,
                command=self._schedule_update_results
            ).pack(side=tk.LEFT, padx=6)

        # ── Right: results ───────────────────────────────────────────────
        right = ttk.LabelFrame(frame, text="Calculated Results", padding=10)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        result_defs = [
            ("V1",       "Phase Voltage V1 (V)"),
            ("Ns",       "Synchronous Speed Ns (rpm)"),
            ("ws",       "Angular Speed ωs (rad/s)"),
            ("T_st",     "Starting Torque T_st (N·m)"),
            ("T_fl",     "Full-Load Torque T_fl (N·m)"),
            ("T_max",    "Maximum Torque T_max (N·m)"),
            ("s_m",      "Slip at Max Torque s_m"),
            ("N_maxT",   "Speed at Max Torque (rpm)"),
            ("ratio_st", "T_st / T_max"),
            ("ratio_fl", "T_fl / T_max"),
            ("R_add",    "Additional Rotor R_add (Ω)"),
        ]

        self._result_labels: dict[str, ttk.Label] = {}
        for i, (key, lbl_text) in enumerate(result_defs):
            ttk.Label(right, text=lbl_text + ":").grid(
                row=i, column=0, sticky="w", pady=3, padx=4)
            val_lbl = ttk.Label(right, text="—", foreground="#1565c0",
                                font=("Arial", 10, "bold"))
            val_lbl.grid(row=i, column=1, sticky="w", padx=12)
            self._result_labels[key] = val_lbl

        self._update_result_labels()

    def _schedule_update_results(self) -> None:
        """Debounce rapid slider events with a short after() delay."""
        if hasattr(self, "_update_job"):
            self.root.after_cancel(self._update_job)
        self._update_job = self.root.after(80, self._update_result_labels)

    def _update_result_labels(self) -> None:
        try:
            p = self.calculate_motor_params()
            mapping = {
                "V1":       f"{p['V1']:.4f}",
                "Ns":       f"{p['Ns']:.1f}",
                "ws":       f"{p['ws']:.4f}",
                "T_st":     f"{p['T_st']:.4f}",
                "T_fl":     f"{p['T_fl']:.4f}",
                "T_max":    f"{p['T_max']:.4f}",
                "s_m":      f"{p['s_m']:.4f}  ({p['s_m']*100:.2f} %)",
                "N_maxT":   f"{p['N_maxT']:.1f}",
                "ratio_st": f"{p['ratio_st']:.4f}",
                "ratio_fl": f"{p['ratio_fl']:.4f}",
                "R_add":    f"{p['R_add']:.6f}",
            }
            for key, val in mapping.items():
                if key in self._result_labels:
                    self._result_labels[key].config(text=val)
        except Exception:
            pass

    # ======================================================================
    # TAB 3 – Torque–Speed Characteristics
    # ======================================================================

    def create_tab_3_torque_speed(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Torque-Speed Characteristics")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Plot area
        fig_frame = ttk.Frame(frame)
        fig_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        fig_frame.grid_rowconfigure(0, weight=1)
        fig_frame.grid_columnconfigure(0, weight=1)

        self._fig3 = Figure(figsize=(11, 6), dpi=80)
        self._ax3  = self._fig3.add_subplot(111)
        self._canvas3 = self._embed_figure(self._fig3, fig_frame)
        self._canvas3.get_tk_widget().bind(
            "<Configure>", lambda _e: self._draw_torque_speed())

        # Controls
        ctrl = ttk.Frame(frame)
        ctrl.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(ctrl, text="Update Plot",
                   command=self._draw_torque_speed).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl, text="Reset Defaults",
                   command=self._reset_torque_speed).pack(side=tk.LEFT, padx=6)

        self._draw_torque_speed()

    def _draw_torque_speed(self) -> None:
        try:
            p   = self.calculate_motor_params()
            ax  = self._ax3
            ax.clear()

            s   = np.linspace(0.001, 1.0, 600)
            ws  = p["ws"]
            V1  = p["V1"]
            Ns  = p["Ns"]
            X2  = p["X2"]

            # Four curves for different R2 values
            r2_set = [0.1, 0.2, 0.5, 1.0]
            colours = ["#1565c0", "#2e7d32", "#c62828", "#e65100"]
            for r2, col in zip(r2_set, colours):
                r_div_s = r2 / s
                T = (3.0 / ws) * V1**2 * r_div_s / (r_div_s**2 + X2**2)
                N = Ns * (1.0 - s)
                lw = 2.5 if r2 == p["R2"] else 1.4
                ax.plot(N, T, color=col, linewidth=lw, label=f"R2 = {r2} Ω")

            # Marked operating points (using current R2)
            def mark(speed, torque, marker, colour, label):
                ax.plot(speed, torque, marker, color=colour,
                        markersize=10, zorder=6)
                ax.annotate(
                    label, (speed, torque),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=8, color=colour,
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", alpha=0.7))

            mark(0,              p["T_st"],  "o", "#000000",
                 f"T_st = {p['T_st']:.1f} N·m")
            mark(Ns*(1-p["s_fl"]), p["T_fl"], "*", "#c62828",
                 f"T_fl = {p['T_fl']:.1f} N·m")
            mark(p["N_maxT"],    p["T_max"], "^", "#2e7d32",
                 f"T_max = {p['T_max']:.1f} N·m")

            ax.set_xlabel("Rotor Speed (rpm)", fontsize=11)
            ax.set_ylabel("Torque (N·m)", fontsize=11)
            ax.set_title("Torque–Speed Characteristics  (3-Phase Induction Motor)",
                         fontsize=12)
            ax.legend(loc="upper left", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, Ns * 1.05)
            ax.set_ylim(bottom=0)

            self._fig3.tight_layout()
            self._canvas3.draw_idle()
        except Exception:
            pass

    def _reset_torque_speed(self) -> None:
        self.VL.set(400.0)
        self.R2.set(0.1)
        self.X2.set(1.0)
        self.turns_ratio.set(4.0)
        self.frequency.set(50.0)
        self.poles.set(4)
        self.slip_fl_pct.set(4.0)
        self._draw_torque_speed()

    # ======================================================================
    # TAB 4 – Fault Current Analysis
    # ======================================================================

    def create_tab_4_fault_current(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Fault Current Analysis")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Plot area
        fig_frame = ttk.Frame(frame)
        fig_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        fig_frame.grid_rowconfigure(0, weight=1)
        fig_frame.grid_columnconfigure(0, weight=1)

        self._fig4  = Figure(figsize=(11, 5), dpi=80)
        self._ax4   = self._fig4.add_subplot(111)
        self._canvas4 = self._embed_figure(self._fig4, fig_frame)
        self._canvas4.get_tk_widget().bind(
            "<Configure>", lambda _e: self._draw_fault_current())

        # Controls
        ctrl = ttk.LabelFrame(frame, text="Fault Parameters", padding=6)
        ctrl.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        self._volt_mult  = tk.DoubleVar(value=1.0)
        self._fault_dur  = tk.DoubleVar(value=0.5)

        for col_off, (lbl, var, lo, hi) in enumerate([
            ("Voltage Multiplier (×)", self._volt_mult, 0.5, 2.0),
            ("Fault Duration (s)",     self._fault_dur, 0.1, 1.0),
        ]):
            base = col_off * 4
            ttk.Label(ctrl, text=lbl).grid(row=0, column=base,   padx=6)
            ttk.Scale(ctrl, variable=var, from_=lo, to=hi,
                      orient=tk.HORIZONTAL, length=200,
                      command=lambda _v: self._draw_fault_current()
                      ).grid(row=0, column=base+1, padx=4)
            ttk.Label(ctrl, textvariable=var, width=6).grid(
                row=0, column=base+2, padx=4)

        # Info strip
        info = ttk.LabelFrame(frame, text="Results", padding=4)
        info.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        self._fault_info: dict[str, ttk.Label] = {}
        for col_i, lbl in enumerate(
                ["Peak Current (A)", "RMS Current (A)",
                 "Time Constant τ (s)", "Impedance Z (Ω)", "Phase Angle φ"]):
            ttk.Label(info, text=lbl + ":").grid(
                row=0, column=col_i*2, padx=8)
            v = ttk.Label(info, text="—", foreground="#1565c0",
                          font=("Arial", 10, "bold"))
            v.grid(row=0, column=col_i*2+1, padx=4)
            self._fault_info[lbl] = v

        self._draw_fault_current()

    def _draw_fault_current(self) -> None:
        try:
            p   = self.calculate_motor_params()
            V1  = p["V1"] * self._volt_mult.get()
            R2  = p["R2"]
            X2  = p["X2"]
            f   = p["f"]
            w   = 2.0 * np.pi * f

            Z   = np.sqrt(R2**2 + X2**2)
            phi = np.arctan2(X2, R2)
            tau = X2 / (w * R2)

            t      = np.linspace(0, self._fault_dur.get(), 3000)
            I_ac   = np.sqrt(2) * V1 / Z * np.sin(w * t + phi)
            I_dc   = -np.sqrt(2) * V1 / Z * np.sin(phi) * np.exp(-t / tau)
            I_tot  = I_ac + I_dc
            env    = np.sqrt(2) * V1 / Z * (
                1 + np.abs(np.sin(phi)) * np.exp(-t / tau))

            I_peak = float(np.max(np.abs(I_tot)))
            I_rms  = float(np.sqrt(np.mean(I_tot**2)))

            ax = self._ax4
            ax.clear()
            ax.plot(t, I_tot, "#1565c0",  lw=1.5, label="Total i(t)")
            ax.plot(t, I_ac,  "#2e7d32", lw=1.0, ls="--",
                    alpha=0.75, label="AC component")
            ax.plot(t, I_dc,  "#c62828", lw=1.0, ls="--",
                    alpha=0.75, label="DC offset")
            ax.plot(t,  env,  "k",       lw=1.0, alpha=0.4, label="Envelope")
            ax.plot(t, -env,  "k",       lw=1.0, alpha=0.4)
            ax.axhline(0, color="black", lw=0.5)

            ax.set_xlabel("Time (s)", fontsize=11)
            ax.set_ylabel("Current (A)", fontsize=11)
            ax.set_title("Startup / Fault Current Waveform  "
                         r"$i(t)=\sqrt{2}V_1/Z\,[\sin(\omega t+\varphi)"
                         r"-\sin\varphi\,e^{-t/\tau}]$",
                         fontsize=11)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)

            self._fig4.tight_layout()
            self._canvas4.draw_idle()

            # Update info strip
            updates = {
                "Peak Current (A)":   f"{I_peak:.3f}",
                "RMS Current (A)":    f"{I_rms:.3f}",
                "Time Constant τ (s)":f"{tau:.5f}",
                "Impedance Z (Ω)":    f"{Z:.4f}",
                "Phase Angle φ":      f"{np.degrees(phi):.2f}°",
            }
            for k, v in updates.items():
                self._fault_info[k].config(text=v)
        except Exception:
            pass

    # ======================================================================
    # TAB 5 – Protection Coordination
    # ======================================================================

    def create_tab_5_protection(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Protection Coordination")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        fig_frame = ttk.Frame(frame)
        fig_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        fig_frame.grid_rowconfigure(0, weight=1)
        fig_frame.grid_columnconfigure(0, weight=1)

        self._fig5  = Figure(figsize=(11, 5), dpi=80)
        self._ax5   = self._fig5.add_subplot(111)
        self._canvas5 = self._embed_figure(self._fig5, fig_frame)
        self._canvas5.get_tk_widget().bind(
            "<Configure>", lambda _e: self._draw_protection())

        ctrl = ttk.LabelFrame(frame, text="Relay Settings", padding=6)
        ctrl.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        self._Is_mult = tk.DoubleVar(value=1.0)
        self._TMS     = tk.DoubleVar(value=0.1)

        for col_off, (lbl, var, lo, hi) in enumerate([
            ("Pickup Current Is (× In)", self._Is_mult, 0.5, 3.0),
            ("Time Multiplier (TMS)",    self._TMS,     0.05, 1.0),
        ]):
            base = col_off * 4
            ttk.Label(ctrl, text=lbl).grid(row=0, column=base,   padx=6)
            ttk.Scale(ctrl, variable=var, from_=lo, to=hi,
                      orient=tk.HORIZONTAL, length=200,
                      command=lambda _v: self._draw_protection()
                      ).grid(row=0, column=base+1, padx=4)
            ttk.Label(ctrl, textvariable=var, width=6).grid(
                row=0, column=base+2, padx=4)

        self._draw_protection()

    def _draw_protection(self) -> None:
        try:
            Is  = self._Is_mult.get()
            TMS = self._TMS.get()
            ax  = self._ax5
            ax.clear()

            I_ratio = np.linspace(1.02, 25, 800)

            curves = {
                "Normal Inverse  (k=0.14, α=0.02)":  (0.14,  0.02),
                "Very Inverse    (k=13.5, α=1)":      (13.5,  1.0),
                "Extremely Inverse (k=80, α=2)":      (80.0,  2.0),
            }
            for lbl, (k, alpha) in curves.items():
                denom = np.clip(I_ratio**alpha - 1, 1e-9, None)
                t_r   = np.clip(TMS * k / denom, 0.01, 200.0)
                ax.loglog(I_ratio * Is, t_r, lw=2, label=lbl)

            # Simplified motor starting curve
            I_s = np.linspace(1.05, 7, 300)
            t_s = np.clip(10.0 / (I_s - 0.9), 0.05, 60.0)
            ax.loglog(I_s, t_s, "k--", lw=2, label="Motor Starting Curve")

            ax.axvline(Is, color="gray", ls=":", lw=1.2,
                       label=f"Pickup Is = {Is:.2f}×In")

            ax.set_xlabel("Current (× In)", fontsize=11)
            ax.set_ylabel("Operating Time (s)", fontsize=11)
            ax.set_title("IEC Inverse-Time Overcurrent Relay Characteristics",
                         fontsize=12)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, which="both", alpha=0.3)
            ax.set_xlim(0.9, 30)
            ax.set_ylim(0.01, 200)

            self._fig5.tight_layout()
            self._canvas5.draw_idle()
        except Exception:
            pass

    # ======================================================================
    # TAB 6 – Speed Controller (PID + Fuzzy sub-tabs)
    # ======================================================================

    def create_tab_6_speed_controller(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Speed Controller")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        sub_nb = ttk.Notebook(frame)
        sub_nb.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        pid_frame   = ttk.Frame(sub_nb)
        fuzzy_frame = ttk.Frame(sub_nb)
        sub_nb.add(pid_frame,   text="PID Controller")
        sub_nb.add(fuzzy_frame, text="Fuzzy Controller")
        for f2 in (pid_frame, fuzzy_frame):
            f2.grid_rowconfigure(0, weight=1)
            f2.grid_columnconfigure(0, weight=1)

        self._create_pid_tab(pid_frame)
        self._create_fuzzy_tab(fuzzy_frame)

    # ── PID sub-tab ────────────────────────────────────────────────────────

    def _create_pid_tab(self, parent: ttk.Frame) -> None:
        fig_frame = ttk.Frame(parent)
        fig_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        fig_frame.grid_rowconfigure(0, weight=1)
        fig_frame.grid_columnconfigure(0, weight=1)

        self._fig6pid  = Figure(figsize=(11, 6), dpi=80)
        self._axes6pid = [self._fig6pid.add_subplot(3, 1, i+1)
                          for i in range(3)]
        self._canvas6pid = self._embed_figure(self._fig6pid, fig_frame)

        ctrl = ttk.LabelFrame(parent, text="PID Parameters", padding=6)
        ctrl.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        self._Kp = tk.DoubleVar(value=2.0)
        self._Ki = tk.DoubleVar(value=0.5)
        self._Kd = tk.DoubleVar(value=0.1)

        for col_off, (lbl, var) in enumerate([
            ("Kp", self._Kp), ("Ki", self._Ki), ("Kd", self._Kd)
        ]):
            base = col_off * 3
            ttk.Label(ctrl, text=lbl + ":").grid(row=0, column=base, padx=6)
            # Range 0.001–20 covers practical motor drive gains
            ttk.Scale(ctrl, variable=var, from_=0.001, to=20.0,
                      orient=tk.HORIZONTAL, length=160
                      ).grid(row=0, column=base+1, padx=4)
            ttk.Label(ctrl, textvariable=var, width=7).grid(
                row=0, column=base+2, padx=4)

        ttk.Button(ctrl, text="Run PID Simulation",
                   command=self._run_pid_simulation
                   ).grid(row=0, column=9, padx=14)

        self._run_pid_simulation()

    def _run_pid_simulation(self) -> None:
        try:
            Kp = self._Kp.get()
            Ki = self._Ki.get()
            Kd = self._Kd.get()

            dt    = 0.01
            t_end = 4.0
            t     = np.arange(0, t_end, dt)

            # Reference speed: 1000 → 1350 rpm at t=0.5 s
            w_ref = np.where(t < 0.5,
                             1000.0 * 2*np.pi/60,
                             1350.0 * 2*np.pi/60)
            # Load torque: 0 → 10 N·m at t=2 s
            T_L   = np.where(t < 2.0, 0.0, 10.0)

            J             = 0.1    # moment of inertia (kg·m²)
            damping_coeff = 0.01   # friction coefficient (N·m·s/rad)
            torque_const  = 2.0    # simplified electrical torque constant

            w       = np.zeros(len(t))
            w[0]    = 1000 * 2*np.pi/60
            Te      = np.zeros(len(t))
            err     = np.zeros(len(t))
            intg    = 0.0
            prev_e  = 0.0

            for i in range(1, len(t)):
                e       = w_ref[i] - w[i-1]
                intg   += e * dt
                intg    = np.clip(intg, -1e4, 1e4)
                deriv   = (e - prev_e) / dt
                u       = np.clip(Kp*e + Ki*intg + Kd*deriv, -200, 200)
                Te[i]   = np.clip(torque_const * u, -500, 500)
                err[i]  = e
                dw      = (Te[i] - T_L[i] - damping_coeff * w[i-1]) / J
                w[i]    = np.clip(w[i-1] + dw*dt, 0, 2500*2*np.pi/60)
                prev_e  = e

            w_rpm     = w   * 60 / (2*np.pi)
            w_ref_rpm = w_ref * 60 / (2*np.pi)
            err_rpm   = err * 60 / (2*np.pi)

            for ax in self._axes6pid:
                ax.clear()

            ax0, ax1, ax2 = self._axes6pid
            ax0.plot(t, w_rpm,     "#1565c0", lw=1.5, label="Actual")
            ax0.plot(t, w_ref_rpm, "r--",     lw=1.5, label="Reference")
            ax0.set_ylabel("Speed (rpm)");  ax0.legend(fontsize=9)
            ax0.set_title(f"PID Response  (Kp={Kp:.2f}  Ki={Ki:.2f}  "
                          f"Kd={Kd:.2f})", fontsize=11)
            ax0.grid(True, alpha=0.3)

            ax1.plot(t, Te,  "#2e7d32", lw=1.5, label="Te")
            ax1.plot(t, T_L, "r--",     lw=1.5, label="T_Load")
            ax1.set_ylabel("Torque (N·m)");  ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)

            ax2.plot(t, err_rpm, "#c62828", lw=1.5, label="Speed error")
            ax2.axhline(0, color="k", lw=0.5)
            ax2.set_xlabel("Time (s)");  ax2.set_ylabel("Error (rpm)")
            ax2.legend(fontsize=9);  ax2.grid(True, alpha=0.3)

            self._fig6pid.tight_layout()
            self._canvas6pid.draw_idle()
        except Exception:
            pass

    # ── Fuzzy sub-tab ──────────────────────────────────────────────────────

    def _create_fuzzy_tab(self, parent: ttk.Frame) -> None:
        fig_frame = ttk.Frame(parent)
        fig_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        fig_frame.grid_rowconfigure(0, weight=1)
        fig_frame.grid_columnconfigure(0, weight=1)

        self._fig6fz   = Figure(figsize=(11, 6), dpi=80)
        self._axes6fz  = [self._fig6fz.add_subplot(2, 1, i+1) for i in range(2)]
        self._canvas6fz = self._embed_figure(self._fig6fz, fig_frame)

        ctrl = ttk.LabelFrame(parent, text="Fuzzy Controller", padding=6)
        ctrl.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        ttk.Button(ctrl, text="Run Fuzzy Simulation",
                   command=self._run_fuzzy_simulation
                   ).pack(side=tk.LEFT, padx=8)
        ttk.Button(ctrl, text="Show Membership Functions",
                   command=self._show_mf_window
                   ).pack(side=tk.LEFT, padx=8)

        self._run_fuzzy_simulation()

    @staticmethod
    def _trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
        """Triangular membership function."""
        out = np.zeros_like(x, dtype=float)
        if b > a:
            out = np.where((x > a) & (x <= b),
                           (x - a) / (b - a), out)
        if c > b:
            out = np.where((x > b) & (x < c),
                           (c - x) / (c - b), out)
        out = np.where(x == b, 1.0, out)
        return np.clip(out, 0.0, 1.0)

    @staticmethod
    def _fuzzy_infer(error: float, d_error: float) -> float:
        """Simple Mamdani fuzzy inference, output in [−1, +1]."""
        e  = np.clip(error,   -1.0, 1.0)
        de = np.clip(d_error, -1.0, 1.0)

        # Membership degrees for 5 linguistic values
        def mf5(v):
            x = np.array([v])
            return (
                float(InductionMotorApp._trimf(x, -1.4, -1.0, -0.5)[0]),
                float(InductionMotorApp._trimf(x, -1.0, -0.5,  0.0)[0]),
                float(InductionMotorApp._trimf(x, -0.5,  0.0,  0.5)[0]),
                float(InductionMotorApp._trimf(x,  0.0,  0.5,  1.0)[0]),
                float(InductionMotorApp._trimf(x,  0.5,  1.0,  1.4)[0]),
            )

        NB_e, NM_e, Z_e, PM_e, PB_e   = mf5(e)
        NB_d, NM_d, Z_d, PM_d, PB_d   = mf5(de)
        crisp = [-1.0, -0.5, 0.0, 0.5, 1.0]

        # 5×5 rule table (row=error, col=dError)
        rule_out = [
            [-1, -1, -.5, -.5,  0],
            [-1, -.5,-.5,  0,  .5],
            [-.5,-.5, 0,  .5,  .5],
            [-.5, 0,  .5, .5,  1],
            [ 0,  .5, .5,  1,   1],
        ]
        e_mfs  = [NB_e, NM_e, Z_e, PM_e, PB_e]
        de_mfs = [NB_d, NM_d, Z_d, PM_d, PB_d]

        num = den = 0.0
        for i, em in enumerate(e_mfs):
            for j, dm in enumerate(de_mfs):
                strength = min(em, dm)
                out_val  = rule_out[i][j]
                num     += strength * out_val
                den     += strength

        return num / den if den > 1e-10 else 0.0

    def _run_fuzzy_simulation(self) -> None:
        try:
            dt    = 0.01
            t_end = 4.0
            t     = np.arange(0, t_end, dt)
            w_ref = np.where(t < 0.5, 1000*2*np.pi/60, 1350*2*np.pi/60)
            T_L   = np.where(t < 2.0, 0.0, 10.0)

            # Shared motor model (same parameters as PID tab, scaled by w_max)
            # Fuzzy output is normalised to [-1,+1], so torque_const is larger
            J             = 0.1    # moment of inertia (kg·m²)
            damping_coeff = 0.01   # friction (N·m·s/rad)
            torque_const  = 60.0   # fuzzy/PID output scale → N·m
            w_max         = 1500 * 2*np.pi/60

            # PID gains are read from the shared DoubleVars
            Kp = self._Kp.get()
            Ki = self._Ki.get()
            Kd = self._Kd.get()

            def simulate_ctrl(use_fuzzy):
                w             = np.zeros(len(t))
                w[0]          = 1000 * 2*np.pi/60
                Te            = np.zeros(len(t))
                intg          = 0.0
                prev_e_norm   = 0.0

                for i in range(1, len(t)):
                    e       = w_ref[i] - w[i-1]
                    e_norm  = np.clip(e / w_max, -1, 1)
                    de_norm = np.clip(
                        (e_norm - prev_e_norm) / dt * 0.05, -1, 1)

                    if use_fuzzy:
                        u = self._fuzzy_infer(e_norm, de_norm)
                    else:
                        intg  += e * dt
                        intg   = np.clip(intg, -1e4, 1e4)
                        de     = (e - prev_e_norm * w_max) / dt
                        u      = np.clip(
                            (Kp*e + Ki*intg + Kd*de) / w_max, -1, 1)

                    Te[i]       = np.clip(torque_const * u, -300, 300)
                    dw          = (Te[i] - T_L[i] - damping_coeff * w[i-1]) / J
                    w[i]        = np.clip(w[i-1] + dw * dt, 0, 2*w_max)
                    prev_e_norm = e_norm
                return w * 60 / (2*np.pi), Te

            w_fz, Te_fz = simulate_ctrl(use_fuzzy=True)
            w_pid, Te_pid = simulate_ctrl(use_fuzzy=False)
            w_ref_rpm = w_ref * 60 / (2*np.pi)

            for ax in self._axes6fz:
                ax.clear()

            ax0, ax1 = self._axes6fz
            ax0.plot(t, w_fz,      "#1565c0", lw=1.5, label="Fuzzy")
            ax0.plot(t, w_pid,     "#2e7d32", lw=1.5, ls="--", label="PID")
            ax0.plot(t, w_ref_rpm, "r-",      lw=1.2, alpha=0.6,
                     label="Reference")
            ax0.set_ylabel("Speed (rpm)");  ax0.legend(fontsize=9)
            ax0.set_title("Fuzzy vs PID Speed Controller Comparison",
                          fontsize=11)
            ax0.grid(True, alpha=0.3)

            ax1.plot(t, Te_fz,  "#1565c0", lw=1.5, label="Fuzzy Torque")
            ax1.plot(t, Te_pid, "#2e7d32", lw=1.5, ls="--",
                     label="PID Torque")
            ax1.plot(t, T_L,   "r--",     lw=1.5, label="Load Torque")
            ax1.set_xlabel("Time (s)");  ax1.set_ylabel("Torque (N·m)")
            ax1.legend(fontsize=9);  ax1.grid(True, alpha=0.3)

            self._fig6fz.tight_layout()
            self._canvas6fz.draw_idle()
        except Exception:
            pass

    def _show_mf_window(self) -> None:
        """Pop-up window showing triangular MFs."""
        try:
            win = tk.Toplevel(self.root)
            win.title("Fuzzy Membership Functions")
            win.geometry("860x440")
            win.grid_rowconfigure(0, weight=1)
            win.grid_columnconfigure(0, weight=1)

            fig = Figure(figsize=(10, 4.5), dpi=90)
            x = np.linspace(-1.2, 1.2, 600)
            labels  = ["NB", "NM", "Z",  "PM", "PB"]
            centres = [-1.0, -0.5, 0.0,  0.5,  1.0]
            colours = ["#1565c0", "#00838f", "#2e7d32", "#e65100", "#c62828"]

            for sp_idx, title in enumerate(["Error (e)", "Rate of Error (Δe)"]):
                ax = fig.add_subplot(1, 2, sp_idx + 1)
                for lbl, ctr, col in zip(labels, centres, colours):
                    ax.plot(x, self._trimf(x, ctr-0.4, ctr, ctr+0.4),
                            color=col, lw=2, label=lbl)
                ax.set_title(title, fontsize=11)
                ax.set_xlabel("Normalized value")
                ax.set_ylabel("Membership degree")
                ax.legend(fontsize=9);  ax.grid(True, alpha=0.3)
                ax.set_ylim(-0.05, 1.1)

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
            canvas.draw()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ======================================================================
    # TAB 7 – Thermal Analysis
    # ======================================================================

    def create_tab_7_thermal(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Thermal Analysis")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        fig_frame = ttk.Frame(frame)
        fig_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        fig_frame.grid_rowconfigure(0, weight=1)
        fig_frame.grid_columnconfigure(0, weight=1)

        self._fig7  = Figure(figsize=(11, 5), dpi=80)
        self._ax7   = self._fig7.add_subplot(111)
        self._canvas7 = self._embed_figure(self._fig7, fig_frame)
        self._canvas7.get_tk_widget().bind(
            "<Configure>", lambda _e: self._draw_thermal())

        ctrl = ttk.LabelFrame(frame, text="Thermal Parameters", padding=6)
        ctrl.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        self._T_amb   = tk.DoubleVar(value=25.0)
        self._overload = tk.DoubleVar(value=1.0)
        self._Rth1    = tk.DoubleVar(value=0.5)
        self._Rth2    = tk.DoubleVar(value=1.0)

        for col_off, (lbl, var, lo, hi) in enumerate([
            ("Ambient Temp (°C)",  self._T_amb,    0,    50),
            ("Overload Factor",    self._overload, 0.5,  2.0),
            ("Rth1 (K/W)",         self._Rth1,    0.1,  2.0),
            ("Rth2 (K/W)",         self._Rth2,    0.1,  5.0),
        ]):
            base = col_off * 4
            ttk.Label(ctrl, text=lbl).grid(row=0, column=base,   padx=6)
            ttk.Scale(ctrl, variable=var, from_=lo, to=hi,
                      orient=tk.HORIZONTAL, length=150,
                      command=lambda _v: self._draw_thermal()
                      ).grid(row=0, column=base+1, padx=4)
            ttk.Label(ctrl, textvariable=var, width=6).grid(
                row=0, column=base+2, padx=4)

        self._draw_thermal()

    def _draw_thermal(self) -> None:
        try:
            p         = self.calculate_motor_params()
            overload  = np.clip(self._overload.get(), 0.1, 5.0)
            T_amb     = self._T_amb.get()
            Rth1      = max(self._Rth1.get(), 1e-3)
            Rth2      = max(self._Rth2.get(), 1e-3)
            Cth1      = 800.0
            Cth2      = 3000.0
            P_core    = 250.0  # fixed core loss (W)

            # Approximate full-load current from T_fl and voltage
            I_fl = p["T_fl"] * p["ws"] / (3 * p["V1"]) if p["V1"] > 0 else 50
            I_fl = np.clip(I_fl, 1, 5000)
            P_cu = 3 * (I_fl * overload)**2 * p["R2"]
            P_loss = P_cu + P_core

            dt = 2.0                       # simulation step: 2 s
            t  = np.arange(0, 3601, dt)   # 0 → 3600 s (1 hour)
            T1 = np.empty(len(t));  T1[0] = T_amb
            T2 = np.empty(len(t));  T2[0] = T_amb

            # Repeated-starting spikes at t≈0, 600 s, 1200 s
            def extra_loss(ti):
                for start in (0, 600, 1200):
                    if start <= ti < start + 30:
                        return P_loss * 6.0
                return 0.0

            for i in range(1, len(t)):
                P_ex   = extra_loss(t[i])
                q12    = (T1[i-1] - T2[i-1]) / Rth1
                q2a    = (T2[i-1] - T_amb)   / Rth2
                dT1    = (P_loss + P_ex - q12) / Cth1
                dT2    = (q12 - q2a) / Cth2
                T1[i]  = T1[i-1] + dT1 * dt
                T2[i]  = T2[i-1] + dT2 * dt

            ax = self._ax7
            ax.clear()
            t_min = t / 60.0
            ax.plot(t_min, T1, "#c62828", lw=2, label="Winding Temp T₁")
            ax.plot(t_min, T2, "#1565c0", lw=2, label="Frame Temp T₂")

            limits = [
                (130, "#e65100", "Class B (130 °C)"),
                (155, "#c62828", "Class F (155 °C)"),
                (180, "#880e4f", "Class H (180 °C)"),
            ]
            for temp, col, lbl in limits:
                ax.axhline(temp, color=col, ls="--", lw=1.5, label=lbl)

            ax.fill_between(t_min, 130, 155, alpha=0.08, color="#e65100")
            ax.fill_between(t_min, 155, 180, alpha=0.08, color="#c62828")
            ax.fill_between(t_min, 180, 220, alpha=0.08, color="#880e4f")

            ax.set_xlabel("Time (minutes)", fontsize=11)
            ax.set_ylabel("Temperature (°C)", fontsize=11)
            ax.set_title(
                f"Motor Thermal Analysis  (Overload = {overload:.1f}×,"
                f"  P_loss = {P_loss:.1f} W)", fontsize=11)
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)

            self._fig7.tight_layout()
            self._canvas7.draw_idle()
        except Exception:
            pass

    # ======================================================================
    # TAB 8 – Economic Analysis
    # ======================================================================

    def create_tab_8_economic(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Economic Analysis")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        fig_frame = ttk.Frame(frame)
        fig_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        fig_frame.grid_rowconfigure(0, weight=1)
        fig_frame.grid_columnconfigure(0, weight=1)

        self._fig8   = Figure(figsize=(11, 4.5), dpi=80)
        self._axes8  = [self._fig8.add_subplot(1, 2, i+1) for i in range(2)]
        self._canvas8 = self._embed_figure(self._fig8, fig_frame)

        ctrl = ttk.LabelFrame(frame, text="Economic Parameters", padding=6)
        ctrl.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        self._elec_price  = tk.DoubleVar(value=0.12)
        self._op_hours    = tk.DoubleVar(value=6000.0)
        self._motor_kw    = tk.DoubleVar(value=15.0)
        self._load_factor = tk.DoubleVar(value=1.0)

        for col_off, (lbl, var, lo, hi) in enumerate([
            ("Electricity ($/kWh)", self._elec_price,  0.05, 0.50),
            ("Op. Hours/Year",      self._op_hours,   1000, 8760),
            ("Motor Rating (kW)",   self._motor_kw,     1,   200),
            ("Load Factor",         self._load_factor, 0.25,  1.25),
        ]):
            base = col_off * 4
            ttk.Label(ctrl, text=lbl).grid(row=0, column=base,   padx=6)
            ttk.Scale(ctrl, variable=var, from_=lo, to=hi,
                      orient=tk.HORIZONTAL, length=150,
                      command=lambda _v: self._draw_economic()
                      ).grid(row=0, column=base+1, padx=4)
            ttk.Label(ctrl, textvariable=var, width=8).grid(
                row=0, column=base+2, padx=4)

        info = ttk.LabelFrame(frame, text="Economic Summary", padding=4)
        info.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        self._econ_info: dict[str, ttk.Label] = {}
        econ_items = [
            "Annual Energy (kWh)", "Annual Cost ($)",
            "CO₂ (kg/yr)", "VFD Savings ($/yr)",
            "Payback (yr)", "Efficiency at FL",
        ]
        for i, lbl in enumerate(econ_items):
            ttk.Label(info, text=lbl + ":").grid(row=0, column=i*2, padx=8)
            v = ttk.Label(info, text="—", foreground="#1565c0",
                          font=("Arial", 10, "bold"))
            v.grid(row=0, column=i*2+1, padx=4)
            self._econ_info[lbl] = v

        self._draw_economic()

    def _draw_economic(self) -> None:
        try:
            price  = self._elec_price.get()
            hours  = self._op_hours.get()
            P_kw   = self._motor_kw.get()
            lf     = np.clip(self._load_factor.get(), 0.01, 3.0)

            loads     = np.array([0.25, 0.5, 0.75, 1.0, 1.25])
            eta_rated = 0.94    # rated efficiency
            pf_rated  = 0.85

            def efficiency(lf_arr):
                P_out = P_kw * lf_arr
                P_cu  = P_kw * 0.03 * lf_arr**2
                P_fe  = P_kw * 0.015
                P_in  = P_out + P_cu + P_fe
                return np.clip(P_out / P_in, 0.05, 0.999)

            eta = efficiency(loads) * 100
            pf  = np.clip(0.50 + 0.40 * loads, 0, 1.0)

            # Actual operating point
            eta_lf = float(efficiency(np.array([lf]))[0])
            P_out_lf = P_kw * lf
            P_in_lf  = P_out_lf / eta_lf
            kwh_yr   = P_in_lf * hours
            cost_yr  = kwh_yr * price
            co2_yr   = kwh_yr * 0.5

            vfd_save = cost_yr * 0.125
            vfd_cost = P_kw * 150
            payback  = vfd_cost / (vfd_save + 0.01)

            for ax in self._axes8:
                ax.clear()

            ax0, ax1 = self._axes8
            ax0.plot(loads*100, eta, "#1565c0", lw=2.0, marker="o",
                     ms=7, label="Efficiency")
            ax0.axvline(lf*100, color="r", ls="--", lw=1.2,
                        label=f"LF={lf:.2f}")
            ax0.set_xlabel("Load (%)");  ax0.set_ylabel("Efficiency (%)")
            ax0.set_title("Efficiency vs Load", fontsize=11)
            ax0.legend(fontsize=9);  ax0.grid(True, alpha=0.3)
            ax0.set_ylim(75, 100)

            ax1.plot(loads*100, pf, "#c62828", lw=2.0, marker="s",
                     ms=7, label="Power Factor")
            ax1.axvline(lf*100, color="b", ls="--", lw=1.2,
                        label=f"LF={lf:.2f}")
            ax1.set_xlabel("Load (%)");  ax1.set_ylabel("Power Factor")
            ax1.set_title("Power Factor vs Load", fontsize=11)
            ax1.legend(fontsize=9);  ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.05)

            self._fig8.tight_layout()
            self._canvas8.draw_idle()

            updates = {
                "Annual Energy (kWh)": f"{kwh_yr:,.1f}",
                "Annual Cost ($)":     f"${cost_yr:,.2f}",
                "CO₂ (kg/yr)":        f"{co2_yr:,.1f}",
                "VFD Savings ($/yr)":  f"${vfd_save:,.2f}",
                "Payback (yr)":        f"{payback:.2f}",
                "Efficiency at FL":    f"{eta_lf*100:.2f} %",
            }
            for k, v in updates.items():
                self._econ_info[k].config(text=v)
        except Exception:
            pass

    # ======================================================================
    # TAB 9 – Harmonic Analysis
    # ======================================================================

    def create_tab_9_harmonic(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Harmonic Analysis")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        fig_frame = ttk.Frame(frame)
        fig_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        fig_frame.grid_rowconfigure(0, weight=1)
        fig_frame.grid_columnconfigure(0, weight=1)

        self._fig9   = Figure(figsize=(11, 4.5), dpi=80)
        self._axes9  = [self._fig9.add_subplot(1, 2, i+1) for i in range(2)]
        self._canvas9 = self._embed_figure(self._fig9, fig_frame)

        ctrl = ttk.LabelFrame(frame, text="Harmonic Content (% of fundamental)",
                              padding=6)
        ctrl.grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        self._h5  = tk.DoubleVar(value=15.0)
        self._h7  = tk.DoubleVar(value=8.0)
        self._h11 = tk.DoubleVar(value=4.0)
        self._h13 = tk.DoubleVar(value=3.0)

        for col_off, (lbl, var, lo, hi) in enumerate([
            ("5th  (%)",  self._h5,  0, 30),
            ("7th  (%)",  self._h7,  0, 20),
            ("11th (%)", self._h11,  0, 15),
            ("13th (%)", self._h13,  0, 10),
        ]):
            base = col_off * 4
            ttk.Label(ctrl, text=lbl).grid(row=0, column=base,   padx=6)
            ttk.Scale(ctrl, variable=var, from_=lo, to=hi,
                      orient=tk.HORIZONTAL, length=140,
                      command=lambda _v: self._draw_harmonic()
                      ).grid(row=0, column=base+1, padx=4)
            ttk.Label(ctrl, textvariable=var, width=5).grid(
                row=0, column=base+2, padx=4)

        info = ttk.LabelFrame(frame, text="Harmonic Analysis Results",
                              padding=4)
        info.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        self._harm_info: dict[str, ttk.Label] = {}
        for i, lbl in enumerate(
                ["THD_V (%)", "Displacement PF", "True PF",
                 "IEEE 519 Limit", "Status"]):
            ttk.Label(info, text=lbl + ":").grid(row=0, column=i*2, padx=8)
            v = ttk.Label(info, text="—", foreground="#1565c0",
                          font=("Arial", 10, "bold"))
            v.grid(row=0, column=i*2+1, padx=4)
            self._harm_info[lbl] = v

        self._draw_harmonic()

    def _draw_harmonic(self) -> None:
        try:
            p   = self.calculate_motor_params()
            V1  = p["V1"]
            f   = p["f"]
            w   = 2 * np.pi * f

            h = {
                5:  self._h5.get()  / 100,
                7:  self._h7.get()  / 100,
                11: self._h11.get() / 100,
                13: self._h13.get() / 100,
            }
            THD = np.sqrt(sum(v**2 for v in h.values())) * 100

            t = np.linspace(0, 2.0/f, 1200)
            v_total = V1 * np.sqrt(2) * np.sin(w * t)
            for n, amp in h.items():
                v_total += amp * V1 * np.sqrt(2) * np.sin(n * w * t)

            disp_pf = 0.85
            true_pf = disp_pf / np.sqrt(1 + (THD/100)**2)

            for ax in self._axes9:
                ax.clear()

            ax0, ax1 = self._axes9
            ax0.plot(t*1000, v_total, "#1565c0", lw=1.5, label="Distorted")
            ax0.plot(t*1000, V1*np.sqrt(2)*np.sin(w*t),
                     "r--", lw=1.2, alpha=0.7, label="Fundamental")
            ax0.set_xlabel("Time (ms)");  ax0.set_ylabel("Voltage (V)")
            ax0.set_title("Voltage Waveform with Harmonics", fontsize=11)
            ax0.legend(fontsize=9);  ax0.grid(True, alpha=0.3)

            orders = [1, 5, 7, 11, 13]
            magnitudes = [100.0, self._h5.get(), self._h7.get(),
                          self._h11.get(), self._h13.get()]
            bar_colours = ["#2e7d32" if m <= 5 else "#c62828"
                           for m in magnitudes]
            bar_colours[0] = "#1565c0"
            ax1.bar([str(o) for o in orders], magnitudes, color=bar_colours,
                    edgecolor="white", linewidth=0.8)
            ax1.axhline(5.0, color="#c62828", ls="--", lw=1.5,
                        label="IEEE 519 limit (5 %)")
            ax1.set_xlabel("Harmonic Order")
            ax1.set_ylabel("Magnitude (% of fundamental)")
            ax1.set_title(f"Harmonic Spectrum  THD = {THD:.2f} %",
                          fontsize=11)
            ax1.legend(fontsize=9);  ax1.grid(True, alpha=0.3, axis="y")

            self._fig9.tight_layout()
            self._canvas9.draw_idle()

            ieee_pass = THD <= 5.0
            self._harm_info["THD_V (%)"].config(text=f"{THD:.2f}")
            self._harm_info["Displacement PF"].config(text=f"{disp_pf:.4f}")
            self._harm_info["True PF"].config(text=f"{true_pf:.4f}")
            self._harm_info["IEEE 519 Limit"].config(text="5.0 %")
            self._harm_info["Status"].config(
                text="PASS ✓" if ieee_pass else "FAIL ✗",
                foreground="#2e7d32" if ieee_pass else "#c62828")
        except Exception:
            pass

    # ======================================================================
    # TAB 10 – Comprehensive Analysis
    # ======================================================================

    def create_tab_10_comprehensive(self) -> None:
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Comprehensive Analysis")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # ── Left: summary text ───────────────────────────────────────────
        left = ttk.LabelFrame(frame, text="Summary of Results", padding=6)
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self._summary_txt = tk.Text(left, wrap=tk.WORD,
                                    font=("Courier", 9), width=52)
        vsb = ttk.Scrollbar(left, orient="vertical",
                             command=self._summary_txt.yview)
        self._summary_txt.configure(yscrollcommand=vsb.set)
        self._summary_txt.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # ── Right: radar chart ───────────────────────────────────────────
        right = ttk.LabelFrame(frame, text="Performance Radar Chart",
                               padding=6)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._fig10 = Figure(figsize=(5.5, 5.5), dpi=90)
        self._ax10  = self._fig10.add_subplot(111, projection="polar")
        self._canvas10 = self._embed_figure(self._fig10, right)

        # ── Buttons ──────────────────────────────────────────────────────
        btn = ttk.Frame(frame)
        btn.grid(row=1, column=0, columnspan=2, pady=8)
        ttk.Button(btn, text="Refresh Analysis",
                   command=self._update_comprehensive
                   ).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn, text="Export Summary",
                   command=self._export_summary
                   ).pack(side=tk.LEFT, padx=10)

        self._update_comprehensive()

    def _update_comprehensive(self) -> None:
        try:
            p = self.calculate_motor_params()
            thd = np.sqrt(
                (self._h5.get()/100)**2 + (self._h7.get()/100)**2 +
                (self._h11.get()/100)**2 + (self._h13.get()/100)**2
            ) * 100

            # ── Summary text ─────────────────────────────────────────────
            self._summary_txt.config(state=tk.NORMAL)
            self._summary_txt.delete("1.0", tk.END)
            lines = [
                "=" * 52,
                "  3-PHASE INDUCTION MOTOR  –  COMPREHENSIVE RESULTS",
                "=" * 52,
                "",
                "MOTOR SPECIFICATIONS",
                "─" * 40,
                f"  Line Voltage VL       : {p['VL']:.2f} V",
                f"  Phase Voltage V1      : {p['V1']:.4f} V",
                f"  Frequency             : {p['f']:.1f} Hz",
                f"  Poles                 : {p['poles']}",
                f"  Rotor Resistance R2   : {p['R2']:.4f} Ω",
                f"  Standstill React. X2  : {p['X2']:.4f} Ω",
                f"  Turns Ratio N1/N2     : {p['n_ratio']:.1f}",
                "",
                "PERFORMANCE RESULTS",
                "─" * 40,
                f"  Synchronous Speed Ns  : {p['Ns']:.1f} rpm",
                f"  Angular Speed ωs      : {p['ws']:.4f} rad/s",
                f"  Starting Torque T_st  : {p['T_st']:.4f} N·m",
                f"  Full-Load Torque T_fl : {p['T_fl']:.4f} N·m",
                f"  Maximum Torque T_max  : {p['T_max']:.4f} N·m",
                f"  Slip at Max Torque    : {p['s_m']:.4f} "
                f"({p['s_m']*100:.2f} %)",
                f"  Speed at Max Torque   : {p['N_maxT']:.1f} rpm",
                "",
                "TORQUE RATIOS",
                "─" * 40,
                f"  T_st / T_max          : {p['ratio_st']:.4f}",
                f"  T_fl / T_max          : {p['ratio_fl']:.4f}",
                "",
                "ADDITIONAL ROTOR RESISTANCE",
                "─" * 40,
                f"  R_add (actual rotor)  : {p['R_add']:.6f} Ω",
                f"                          {p['R_add']*1000:.4f} mΩ",
                "",
                "ECONOMIC PARAMETERS",
                "─" * 40,
                f"  Motor Rating          : {self._motor_kw.get():.1f} kW",
                f"  Op. Hours / Year      : {self._op_hours.get():.0f} h",
                f"  Electricity Price     : ${self._elec_price.get():.3f}/kWh",
                "",
                "HARMONIC CONTENT",
                "─" * 40,
                f"  5th  Harmonic         : {self._h5.get():.1f} %",
                f"  7th  Harmonic         : {self._h7.get():.1f} %",
                f"  11th Harmonic         : {self._h11.get():.1f} %",
                f"  13th Harmonic         : {self._h13.get():.1f} %",
                f"  THD                   : {thd:.2f} %",
                "=" * 52,
            ]
            self._summary_txt.insert(tk.END, "\n".join(lines))
            self._summary_txt.config(state=tk.DISABLED)

            # ── Radar chart ──────────────────────────────────────────────
            ax = self._ax10
            ax.clear()

            categories = [
                "T_st\n(norm.)", "T_fl\n(norm.)", "T_max\n(norm.)",
                "Efficiency", "Power\nFactor", "THD\n(inv.)",
            ]
            N = len(categories)

            vals = [
                np.clip(p["T_st"]  / 550.0, 0, 1),
                np.clip(p["T_fl"]  / 700.0, 0, 1),
                np.clip(p["T_max"] / 900.0, 0, 1),
                0.93,   # approximate efficiency
                0.85,   # approximate power factor
                np.clip(1 - thd / 100 * 4, 0, 1),  # inverted THD
            ]

            angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
            vals_plot   = vals   + [vals[0]]
            angles_plot = angles + [angles[0]]

            ax.plot(angles_plot,  vals_plot, "#1565c0", lw=2, marker="o", ms=6)
            ax.fill(angles_plot,  vals_plot, alpha=0.22, color="#1565c0")
            ax.set_xticks(angles)
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
            ax.set_title("Performance Metrics\n(Normalized)", pad=20,
                         fontsize=10)
            ax.grid(True)

            self._fig10.tight_layout()
            self._canvas10.draw_idle()
        except Exception:
            pass

    def _export_summary(self) -> None:
        try:
            p = self.calculate_motor_params()
            msg = (
                f"3-Phase Induction Motor – Analysis Summary\n"
                f"{'─'*48}\n"
                f"VL = {p['VL']:.0f} V   V1 = {p['V1']:.2f} V   "
                f"f = {p['f']:.0f} Hz   Poles = {p['poles']}\n"
                f"R2 = {p['R2']:.4f} Ω   X2 = {p['X2']:.4f} Ω\n"
                f"T_st  = {p['T_st']:.4f} N·m\n"
                f"T_fl  = {p['T_fl']:.4f} N·m\n"
                f"T_max = {p['T_max']:.4f} N·m\n"
                f"s_m   = {p['s_m']:.4f}   N_maxT = {p['N_maxT']:.0f} rpm\n"
                f"T_st/T_max = {p['ratio_st']:.4f}   "
                f"T_fl/T_max = {p['ratio_fl']:.4f}\n"
                f"R_add = {p['R_add']:.6f} Ω  ({p['R_add']*1000:.4f} mΩ)\n"
            )
            messagebox.showinfo("Export Summary", msg)
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app  = InductionMotorApp(root)
    root.mainloop()
