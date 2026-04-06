"""wound_rotor_dfim_suite.py
==========================
Wound-Rotor Induction Motor with External Rotor Injection (Doubly-Fed)
Complete Engineering Analysis Suite – Python / Tkinter

PROBLEM STATEMENT
-----------------
The stator of a 2-pole wound-rotor motor is powered by a 3-phase, 50 Hz
source.  The rotor is connected to a 3-phase, 11 Hz source.

  (1) At what speeds can the rotor turn?
  (2) If the motor operates at the LOWER speed, does the 11 Hz source
      absorb or deliver active power?

SOLUTION SUMMARY
----------------
  Ns  = 120 × 50 / 2 = 3 000 rpm

  The slip frequency must equal the rotor-injection frequency:
      s × f_s = f_r  →  s = ±11/50 = ±0.22

  Sub-synchronous  (s = +0.22): N₁ = 3000 × 0.78 = 2 340 rpm  ← LOWER
  Super-synchronous (s = −0.22): N₂ = 3000 × 1.22 = 3 660 rpm ← HIGHER

  At the lower speed (s = +0.22):
      Slip power = s × P_air-gap  > 0  →  rotor source ABSORBS power
      (Scherbius / slip-power-recovery principle)
"""

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helper: embed a Figure in a tk parent widget
# ─────────────────────────────────────────────────────────────────────────────
def _embed(fig: Figure, parent: tk.Widget) -> FigureCanvasTkAgg:
    canvas = FigureCanvasTkAgg(fig, master=parent)
    widget = canvas.get_tk_widget()
    widget.grid(row=0, column=0, sticky="nsew")
    parent.grid_rowconfigure(0, weight=1)
    parent.grid_columnconfigure(0, weight=1)
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────
class DFIMApp:
    """Doubly-Fed Induction Motor (Wound-Rotor) Analysis Suite."""

    # ── Construction ──────────────────────────────────────────────────────
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(
            "Wound-Rotor Motor with Rotor Injection – Engineering Suite"
        )
        self.root.geometry("1420x960")
        self.root.resizable(True, True)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # ── Motor parameters (tk variables) ───────────────────────────────
        self.VL        = tk.DoubleVar(value=400.0)   # line voltage V
        self.f_s       = tk.DoubleVar(value=50.0)    # stator freq Hz
        self.f_r       = tk.DoubleVar(value=11.0)    # rotor inject freq Hz
        self.poles     = tk.DoubleVar(value=2.0)     # (cast to int in _p)
        self.R1        = tk.DoubleVar(value=0.5)     # stator resistance Ω
        self.X1        = tk.DoubleVar(value=1.0)     # stator reactance Ω
        self.R2        = tk.DoubleVar(value=0.3)     # rotor resistance Ω
        self.X2        = tk.DoubleVar(value=0.8)     # rotor reactance Ω
        self.Xm        = tk.DoubleVar(value=30.0)    # magnetising react Ω

        # ── Controller / simulation knobs ─────────────────────────────────
        self.T_load    = tk.DoubleVar(value=50.0)
        self.Kp        = tk.DoubleVar(value=2.0)
        self.Ki        = tk.DoubleVar(value=0.5)
        self.Kd        = tk.DoubleVar(value=0.1)
        self.N_ref     = tk.DoubleVar(value=2340.0)
        self.T_step    = tk.DoubleVar(value=30.0)
        self.t_step    = tk.DoubleVar(value=2.0)
        self._ctrl_type = tk.StringVar(value="PID")

        # ── Fault knobs ───────────────────────────────────────────────────
        self._fault_t  = tk.DoubleVar(value=0.1)
        self._fault_Zf = tk.DoubleVar(value=0.0)

        # ── Protection knobs ──────────────────────────────────────────────
        self._p_CTI    = tk.DoubleVar(value=0.3)
        self._p_TMS1   = tk.DoubleVar(value=0.1)
        self._p_TMS2   = tk.DoubleVar(value=0.4)

        # ── Thermal knobs ─────────────────────────────────────────────────
        self._Rth      = tk.DoubleVar(value=0.008)   # °C/W  (realistic for ~15 kW motor)
        self._Cth      = tk.DoubleVar(value=500.0)
        self._Ta       = tk.DoubleVar(value=40.0)
        self._kl       = tk.DoubleVar(value=0.8)

        # ── Economic knobs ────────────────────────────────────────────────
        self._tariff   = tk.DoubleVar(value=0.15)
        self._hours    = tk.DoubleVar(value=6000.0)
        self._Prated_kW = tk.DoubleVar(value=15.0)

        # ── Harmonic knobs ────────────────────────────────────────────────
        self._h_order  = tk.DoubleVar(value=25.0)
        self._h_dpf    = tk.DoubleVar(value=0.85)

        # ── Notebook ──────────────────────────────────────────────────────
        nb = ttk.Notebook(self.root)
        nb.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.nb = nb

        self._build_tab_overview()
        self._build_tab_inputs()
        self._build_tab_torque_speed()
        self._build_tab_fault()
        self._build_tab_protection()
        self._build_tab_speed_ctrl()
        self._build_tab_thermal()
        self._build_tab_economic()
        self._build_tab_harmonic()
        self._build_tab_comprehensive()

    # ── Core calculation ──────────────────────────────────────────────────
    def _p(self) -> dict:
        """Return dict of all derived quantities (safe, no exceptions)."""
        VL   = float(self.VL.get())
        f_s  = max(float(self.f_s.get()), 1.0)
        f_r  = max(float(self.f_r.get()), 0.1)
        pol  = max(int(round(float(self.poles.get()))), 2)
        R1   = max(float(self.R1.get()), 1e-6)
        X1   = max(float(self.X1.get()), 1e-6)
        R2   = max(float(self.R2.get()), 1e-6)
        X2   = max(float(self.X2.get()), 1e-6)
        Xm   = max(float(self.Xm.get()), 1e-6)

        V1   = VL / np.sqrt(3.0)
        Ns   = 120.0 * f_s / pol
        ws   = 2.0 * np.pi * Ns / 60.0          # rad/s (synchronous)

        s_sub = f_r / f_s          # +0.22  subsynchronous
        s_sup = -f_r / f_s         # -0.22  supersynchronous
        N_sub = Ns * (1.0 - s_sub)
        N_sup = Ns * (1.0 - s_sup)

        def _torque(s: float) -> float:
            denom = (R1 + R2 / s) ** 2 + (X1 + X2) ** 2
            return (3.0 / ws) * V1 ** 2 * (R2 / s) / denom

        T_sub = _torque(s_sub)
        T_sup = _torque(abs(s_sup) if s_sup < 0 else s_sup)
        # For supersynchronous, use |s| in magnitude but negative slip
        denom_sup = (R1 - R2 / abs(s_sup)) ** 2 + (X1 + X2) ** 2
        if denom_sup > 1e-9:
            T_sup = (3.0 / ws) * V1 ** 2 * (-R2 / s_sup) / denom_sup
        else:
            T_sup = 0.0

        # Air-gap power
        P_ag_sub = T_sub * ws
        P_slip    = s_sub * P_ag_sub          # > 0 → source absorbs
        P_mech_sub = (1.0 - s_sub) * P_ag_sub

        P_ag_sup  = T_sup * ws
        P_slip_sup = s_sup * P_ag_sup         # < 0 → source delivers

        return dict(
            VL=VL, f_s=f_s, f_r=f_r, pol=pol,
            R1=R1, X1=X1, R2=R2, X2=X2, Xm=Xm,
            V1=V1, Ns=Ns, ws=ws,
            s_sub=s_sub, s_sup=s_sup,
            N_sub=N_sub, N_sup=N_sup,
            T_sub=T_sub, T_sup=T_sup,
            P_ag_sub=P_ag_sub, P_slip=P_slip, P_mech_sub=P_mech_sub,
            P_ag_sup=P_ag_sup, P_slip_sup=P_slip_sup,
        )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 1 – Overview & Problem Solution
    # ─────────────────────────────────────────────────────────────────────
    def _build_tab_overview(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Overview & Solution  ")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=3)
        tab.grid_columnconfigure(1, weight=2)

        ttk.Label(
            tab,
            text="Wound-Rotor Motor  ·  External Rotor Injection  ·  Engineering Analysis",
            font=("Arial", 13, "bold"),
        ).grid(row=0, column=0, columnspan=2, pady=8)

        # ── Left: explanation text ──────────────────────────────────────
        lf = ttk.LabelFrame(tab, text="Problem Statement & Full Solution", padding=8)
        lf.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        lf.grid_rowconfigure(0, weight=1)
        lf.grid_columnconfigure(0, weight=1)

        txt = tk.Text(lf, wrap="word", font=("Courier New", 10), bg="#f9f9f9",
                      relief="flat", bd=0)
        sb  = ttk.Scrollbar(lf, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        explanation = (
            "PROBLEM\n"
            "═══════════════════════════════════════════════════════\n"
            "The stator of a 2-pole wound-rotor induction motor is\n"
            "powered by a 3-phase, 50 Hz source.  The rotor is\n"
            "connected to a 3-phase, 11 Hz external source.\n\n"
            "  Q1. At what speeds can the rotor turn?\n"
            "  Q2. At the lower speed, does the 11 Hz source absorb\n"
            "      or deliver active power?\n\n"
            "═══════════════════════════════════════════════════════\n"
            "STEP 1 – Synchronous Speed\n"
            "───────────────────────────────────────────────────────\n"
            "  Ns = 120 × f_stator / poles\n"
            "     = 120 × 50 / 2\n"
            "     = 3 000 rpm\n\n"
            "STEP 2 – Slip-Frequency Condition\n"
            "───────────────────────────────────────────────────────\n"
            "  The rotor EMF has frequency = s × f_stator.\n"
            "  For steady-state operation with an external source\n"
            "  at f_r the injection frequency must match:\n\n"
            "       s × f_s  =  f_r\n"
            "       s × 50   =  11\n\n"
            "  The injected source may have the SAME or OPPOSITE\n"
            "  phase sequence as the rotor EMF → two solutions:\n\n"
            "    Case 1 (same seq.):     s = +11/50 = +0.22\n"
            "    Case 2 (opposite seq.): s = -11/50 = -0.22\n\n"
            "STEP 3 – Rotor Speeds\n"
            "───────────────────────────────────────────────────────\n"
            "  N = Ns × (1 - s)\n\n"
            "  Case 1 – Sub-synchronous:\n"
            "    N₁ = 3000 × (1 - 0.22) = 3000 × 0.78 = 2 340 rpm\n"
            "       ← LOWER SPEED  (s > 0)\n\n"
            "  Case 2 – Super-synchronous:\n"
            "    N₂ = 3000 × (1-(-0.22)) = 3000 × 1.22 = 3 660 rpm\n"
            "       ← HIGHER SPEED (s < 0)\n\n"
            "  ∴  The rotor can turn at 2 340 rpm or 3 660 rpm.\n\n"
            "STEP 4 – Power Analysis at Lower Speed (2 340 rpm)\n"
            "───────────────────────────────────────────────────────\n"
            "  s = +0.22 (positive → sub-synchronous motoring)\n\n"
            "  Power balance in the rotor circuit:\n"
            "    P_air-gap  = T_e × ωs                  [W]\n"
            "    P_mech     = (1 - s) × P_ag  (shaft)   [W]\n"
            "    P_slip     = s × P_ag                  [W]\n\n"
            "  With an ideal 11 Hz source connected to the rotor,\n"
            "  it RECEIVES the slip power:\n"
            "    P_ext = s × P_ag = 0.22 × P_ag  > 0\n\n"
            "  ∴  The 11 Hz source ABSORBS active power.\n\n"
            "  This is the SCHERBIUS (slip-power-recovery) principle:\n"
            "  instead of dissipating slip energy as heat in external\n"
            "  rotor resistances, it is recovered by the 11 Hz source\n"
            "  and fed back to the supply.\n\n"
            "STEP 5 – Higher Speed (3 660 rpm, s = −0.22)\n"
            "───────────────────────────────────────────────────────\n"
            "  s < 0 → super-synchronous.\n"
            "  P_ext = s × P_ag = −0.22 × P_ag  < 0\n"
            "  ∴  The 11 Hz source DELIVERS power into the rotor,\n"
            "     adding to the mechanical output.  This is the\n"
            "     Doubly-Fed Induction Generator (DFIG) principle\n"
            "     used in variable-speed wind turbines.\n\n"
            "EQUIVALENT CIRCUIT (per-phase, referred to stator)\n"
            "───────────────────────────────────────────────────────\n"
            "                  R1    jX1          R2/s    jX2\n"
            "  V₁ ──┬── [ ─── R1 ─ jX1 ─ ]─┬─[ R2/s  jX2 ]─┬─ V_inj/s\n"
            "       │                        │                │\n"
            "      jXm                    (air gap)       (rotor)\n"
            "       │                        │                │\n"
            "  ─────┴────────────────────────┴────────────────┴─────────\n\n"
            "  V_inj = injected rotor voltage (11 Hz external source)\n"
            "  At operating point: V_inj chosen so s × f_s = f_r exactly.\n"
            "═══════════════════════════════════════════════════════\n"
        )
        txt.insert("1.0", explanation)
        txt.configure(state="disabled")

        # ── Right: computed results panel ──────────────────────────────
        rf = ttk.LabelFrame(tab, text="Computed Results", padding=10)
        rf.grid(row=1, column=1, sticky="nsew", padx=8, pady=4)
        rf.grid_columnconfigure(1, weight=1)

        fields = [
            ("Stator Frequency  f_s",    "f_s",     "Hz"),
            ("Rotor Inject. Freq  f_r",  "f_r",     "Hz"),
            ("Poles",                    "pol",     ""),
            ("Synchronous Speed  Ns",    "Ns",      "rpm"),
            ("── Subsynchronous ──",     None,      ""),
            ("  Slip  s_sub",            "s_sub",   "pu"),
            ("  Speed  N_sub",           "N_sub",   "rpm"),
            ("  Torque  T_sub",          "T_sub",   "N·m"),
            ("  Air-gap Power",          "P_ag_sub","W"),
            ("  Slip Power (absorbed)",  "P_slip",  "W  ← ABSORBS"),
            ("── Supersynchronous ──",   None,      ""),
            ("  Slip  s_sup",            "s_sup",   "pu"),
            ("  Speed  N_sup",           "N_sup",   "rpm"),
            ("  Torque  T_sup",          "T_sup",   "N·m"),
            ("  Slip Power (delivered)", "P_slip_sup", "W  ← DELIVERS"),
        ]

        self._ov_lbl: dict[str, ttk.Label] = {}
        for i, (label, key, unit) in enumerate(fields):
            if key is None:
                ttk.Separator(rf, orient="horizontal").grid(
                    row=i, column=0, columnspan=3, sticky="ew", pady=3)
                ttk.Label(rf, text=label, font=("Arial", 9, "italic"),
                          foreground="navy").grid(
                    row=i, column=0, columnspan=3, sticky="w")
                continue
            ttk.Label(rf, text=label + ":", anchor="e").grid(
                row=i, column=0, sticky="e", padx=4, pady=2)
            lbl = ttk.Label(rf, text="—", font=("Courier New", 10, "bold"),
                            foreground="#1a237e")
            lbl.grid(row=i, column=1, sticky="w", padx=4)
            ttk.Label(rf, text=unit, foreground="#555").grid(
                row=i, column=2, sticky="w")
            self._ov_lbl[key] = lbl

        ttk.Button(rf, text="↺  Refresh Results",
                   command=self._update_overview).grid(
            row=len(fields)+1, column=0, columnspan=3, pady=12)

        self._update_overview()

    def _update_overview(self) -> None:
        p = self._p()
        fmt = {
            "f_s":      f"{p['f_s']:.1f}",
            "f_r":      f"{p['f_r']:.1f}",
            "pol":      f"{p['pol']}",
            "Ns":       f"{p['Ns']:.1f}",
            "s_sub":    f"{p['s_sub']:.4f}",
            "N_sub":    f"{p['N_sub']:.1f}",
            "T_sub":    f"{p['T_sub']:.2f}",
            "P_ag_sub": f"{p['P_ag_sub']:.1f}",
            "P_slip":   f"{p['P_slip']:.1f}",
            "s_sup":    f"{p['s_sup']:.4f}",
            "N_sup":    f"{p['N_sup']:.1f}",
            "T_sup":    f"{p['T_sup']:.2f}",
            "P_slip_sup": f"{p['P_slip_sup']:.1f}",
        }
        for k, v in fmt.items():
            if k in self._ov_lbl:
                self._ov_lbl[k].configure(text=v)

    # ─────────────────────────────────────────────────────────────────────
    # TAB 2 – Input Parameters
    # ─────────────────────────────────────────────────────────────────────
    def _build_tab_inputs(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Input Parameters  ")
        tab.grid_rowconfigure(1, weight=1)
        for c in range(2):
            tab.grid_columnconfigure(c, weight=1)

        # ── Supply & rotor injection ──────────────────────────────────
        sf = ttk.LabelFrame(tab, text="Supply & Rotor Injection", padding=10)
        sf.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self._add_slider_frame(sf, [
            ("Line Voltage  VL (V)",           self.VL,    100,  1000, ".1f"),
            ("Stator Frequency  f_s (Hz)",      self.f_s,   10,   100, ".1f"),
            ("Rotor Injection  f_r (Hz)",        self.f_r,   1,    50,  ".1f"),
            ("No. of Poles",                     self.poles, 2,    12,  ".0f"),
        ])

        # ── Machine impedances ────────────────────────────────────────
        mf = ttk.LabelFrame(tab, text="Machine Impedances (Ω, per-phase referred to stator)",
                             padding=10)
        mf.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self._add_slider_frame(mf, [
            ("Stator Resistance  R1 (Ω)",  self.R1, 0.01, 5.0,  ".3f"),
            ("Stator Reactance   X1 (Ω)",  self.X1, 0.1,  10.0, ".2f"),
            ("Rotor Resistance   R2 (Ω)",  self.R2, 0.01, 5.0,  ".3f"),
            ("Rotor Reactance    X2 (Ω)",  self.X2, 0.1,  10.0, ".2f"),
            ("Magnetising React. Xm (Ω)",  self.Xm, 5.0,  100,  ".1f"),
        ])

        # ── Live summary ──────────────────────────────────────────────
        sf2 = ttk.LabelFrame(tab, text="Live Calculation Summary", padding=8)
        sf2.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
        sf2.grid_columnconfigure(0, weight=1)

        self._inp_summary = tk.StringVar()
        ttk.Label(sf2, textvariable=self._inp_summary,
                  font=("Courier New", 10)).grid(sticky="w")

        ttk.Button(tab, text="Apply & Refresh All Tabs",
                   command=self._apply_all).grid(
            row=2, column=0, columnspan=2, pady=10)

        # Trace all params to update summary
        for v in (self.VL, self.f_s, self.f_r, self.poles,
                  self.R1, self.X1, self.R2, self.X2, self.Xm):
            v.trace_add("write", lambda *_: self._refresh_input_summary())
        self._refresh_input_summary()

    def _add_slider_frame(
        self,
        parent: tk.Widget,
        specs: list,
    ) -> None:
        """Create a grid of labeled sliders inside *parent*."""
        parent.grid_columnconfigure(0, weight=1)
        for row_idx, (label, var, lo, hi, fmt) in enumerate(specs):
            # Use default-arg binding to capture loop variables safely
            val_str = tk.StringVar(value=f"{var.get():{fmt}}")

            ttk.Label(parent, text=label).grid(
                row=2 * row_idx, column=0, columnspan=3,
                sticky="w", pady=(8, 0))

            def _cmd(v, vs=val_str, f=fmt):
                try:
                    vs.set(f"{float(v):{f}}")
                except Exception:
                    pass

            sl = ttk.Scale(parent, from_=lo, to=hi, variable=var,
                           orient="horizontal", command=_cmd)
            sl.grid(row=2 * row_idx + 1, column=0, sticky="ew", padx=4)

            ttk.Label(parent, textvariable=val_str,
                      font=("Courier New", 10, "bold"),
                      width=9, anchor="e").grid(
                row=2 * row_idx + 1, column=1, padx=4)

            var.trace_add(
                "write",
                lambda *_, vs=val_str, vv=var, f=fmt: self._safe_update_strvar(vs, vv, f),
            )

    @staticmethod
    def _safe_update_strvar(sv: tk.StringVar, var: tk.Variable, fmt: str) -> None:
        try:
            sv.set(f"{float(var.get()):{fmt}}")
        except Exception:
            pass

    def _refresh_input_summary(self) -> None:
        try:
            p = self._p()
            self._inp_summary.set(
                f"Ns = {p['Ns']:.0f} rpm  |  "
                f"Sub-sync: s={p['s_sub']:.3f} → N={p['N_sub']:.0f} rpm  |  "
                f"Super-sync: s={p['s_sup']:.3f} → N={p['N_sup']:.0f} rpm  |  "
                f"V₁ = {p['V1']:.1f} V"
            )
        except Exception:
            pass

    def _apply_all(self) -> None:
        self._update_overview()
        self._refresh_input_summary()
        messagebox.showinfo(
            "Parameters Applied",
            "All parameters updated.\n"
            "Switch to each tab and press its plot/run button to refresh.",
        )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 3 – Torque-Speed Curves
    # ─────────────────────────────────────────────────────────────────────
    def _build_tab_torque_speed(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Torque-Speed  ")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        ctrl = ttk.Frame(tab)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=4)

        ttk.Label(ctrl, text="Load Torque (N·m):").pack(side="left", padx=4)
        ttk.Scale(ctrl, from_=0, to=200, variable=self.T_load,
                  orient="horizontal", length=150).pack(side="left")
        self._tl_lbl = ttk.Label(ctrl, text=f"{self.T_load.get():.0f}",
                                  width=5)
        self._tl_lbl.pack(side="left")
        self.T_load.trace_add(
            "write",
            lambda *_: self._tl_lbl.configure(text=f"{self.T_load.get():.0f}")
        )

        ttk.Button(ctrl, text="Plot Torque-Speed",
                   command=self._plot_ts).pack(side="left", padx=12)

        canv_frame = ttk.Frame(tab)
        canv_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        canv_frame.grid_rowconfigure(0, weight=1)
        canv_frame.grid_columnconfigure(0, weight=1)

        self._fig_ts = Figure(tight_layout=True)
        self._c_ts   = _embed(self._fig_ts, canv_frame)

        self._ts_info = tk.StringVar()
        ttk.Label(tab, textvariable=self._ts_info,
                  font=("Courier New", 10)).grid(row=2, column=0, pady=4)

        self._plot_ts()

    def _plot_ts(self) -> None:
        p = self._p()
        Ns = p["Ns"]; ws = p["ws"]
        V1 = p["V1"]; R1 = p["R1"]; R2 = p["R2"]
        X1 = p["X1"]; X2 = p["X2"]

        # Slip range: -0.6 to +1.5 avoiding zero
        s_arr = np.concatenate([
            np.linspace(-0.6, -1e-3, 400),
            np.linspace(1e-3,  1.5, 600),
        ])
        N_arr = Ns * (1.0 - s_arr)

        def _Te(s):
            d = (R1 + R2 / s) ** 2 + (X1 + X2) ** 2
            return (3.0 / ws) * V1 ** 2 * (R2 / s) / d

        T_arr = np.array([_Te(s) for s in s_arr])

        self._fig_ts.clear()
        ax = self._fig_ts.add_subplot(111)

        ax.plot(N_arr, T_arr, "b-", lw=2, label="Electromagnetic Torque")
        ax.axhline(self.T_load.get(), color="red", ls="--", lw=1.5,
                   label=f"Load Torque = {self.T_load.get():.0f} N·m")
        ax.axvline(Ns, color="gray", ls=":", lw=1, label=f"Ns = {Ns:.0f} rpm")

        for N_op, s_op, T_op, col, lbl in [
            (p["N_sub"], p["s_sub"], p["T_sub"], "green",
             f"Sub-sync  {p['N_sub']:.0f} rpm  (s = {p['s_sub']:.2f})"),
            (p["N_sup"], p["s_sup"], p["T_sup"], "purple",
             f"Super-sync {p['N_sup']:.0f} rpm  (s = {p['s_sup']:.2f})"),
        ]:
            ax.axvline(N_op, color=col, ls="--", lw=1.5, label=lbl)
            ax.scatter([N_op], [T_op], s=120, color=col, zorder=6)
            ax.annotate(f"{T_op:.1f} N·m",
                        xy=(N_op, T_op),
                        xytext=(N_op + 50, T_op + 5),
                        fontsize=9, color=col,
                        arrowprops=dict(arrowstyle="->", color=col))

        ax.set_xlabel("Rotor Speed (rpm)", fontsize=11)
        ax.set_ylabel("Torque (N·m)", fontsize=11)
        ax.set_title(
            "Torque-Speed Characteristic\n"
            "Wound-Rotor Motor with External 11 Hz Rotor Injection",
            fontsize=12,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(Ns * (-0.65), Ns * 1.65)

        self._fig_ts.tight_layout()
        self._c_ts.draw_idle()

        T_max = (3.0 / ws) * V1 ** 2 / (2.0 * X2)
        s_max = R2 / X2
        self._ts_info.set(
            f"Tmax = {T_max:.1f} N·m  at  s = {s_max:.4f}  "
            f"({Ns*(1-s_max):.0f} rpm)"
        )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 4 – Fault Current Simulation
    # ─────────────────────────────────────────────────────────────────────
    def _build_tab_fault(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Fault Current  ")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        ctrl = ttk.Frame(tab)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=4)

        ttk.Label(ctrl, text="Fault Inception (s):").pack(side="left", padx=4)
        ttk.Scale(ctrl, from_=0, to=0.5, variable=self._fault_t,
                  orient="horizontal", length=100).pack(side="left")
        ttk.Label(ctrl, text="Fault Impedance Zf (Ω):").pack(side="left", padx=8)
        ttk.Scale(ctrl, from_=0, to=2.0, variable=self._fault_Zf,
                  orient="horizontal", length=100).pack(side="left")
        ttk.Button(ctrl, text="Simulate Fault",
                   command=self._plot_fault).pack(side="left", padx=10)

        canv_frame = ttk.Frame(tab)
        canv_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        canv_frame.grid_rowconfigure(0, weight=1)
        canv_frame.grid_columnconfigure(0, weight=1)

        self._fig_fc = Figure(tight_layout=True)
        self._c_fc   = _embed(self._fig_fc, canv_frame)

        self._fc_info = tk.StringVar()
        ttk.Label(tab, textvariable=self._fc_info,
                  font=("Courier New", 10), foreground="darkred").grid(
            row=2, column=0, pady=4)

        self._plot_fault()

    def _plot_fault(self) -> None:
        p   = self._p()
        V1  = p["V1"]
        R1, X1, R2, X2 = p["R1"], p["X1"], p["R2"], p["X2"]
        f_s = p["f_s"]
        w   = 2.0 * np.pi * f_s
        Zf  = max(float(self._fault_Zf.get()), 0.0)
        t_f = float(self._fault_t.get())

        # Sub-transient fault impedance (locked rotor)
        Rtot = R1 + R2 + Zf
        Xtot = X1 + X2
        Z_sub = complex(Rtot, Xtot)
        I_rms   = V1 / abs(Z_sub)
        I_peak  = I_rms * np.sqrt(2.0)
        XR      = Xtot / max(Rtot, 1e-9)
        tau_dc  = Xtot / max(w * Rtot, 1e-9)        # DC-offset time constant

        # Worst-case inception angle for maximum asymmetry
        alpha   = np.arctan2(Xtot, Rtot)             # = arctan(X/R)
        theta0  = np.pi / 2.0 - alpha
        I_dc0   = I_peak * np.sin(theta0)             # initial DC offset
        I_peak_asym = I_peak + abs(I_dc0)

        t = np.linspace(0.0, 0.35, 4000)

        # Pre-fault no-load current
        Xm   = p["Xm"]
        Z_nl = complex(R1, X1 + Xm)
        I_nl = V1 / abs(Z_nl)
        i_pre = I_nl * np.sqrt(2.0) * np.sin(w * t)

        # Fault current = AC + DC offset
        i_ac = I_peak * np.sin(w * (t - t_f) + theta0)
        i_dc = -I_dc0 * np.exp(-(t - t_f) / max(tau_dc, 1e-9))
        i_fault = i_ac + i_dc

        i_total = np.where(t < t_f, i_pre, i_fault)

        # Three-phase
        i_B = np.where(t < t_f,
                       I_nl * np.sqrt(2.0) * np.sin(w * t - 2*np.pi/3),
                       I_peak * np.sin(w*(t-t_f) + theta0 - 2*np.pi/3))
        i_C = np.where(t < t_f,
                       I_nl * np.sqrt(2.0) * np.sin(w * t + 2*np.pi/3),
                       I_peak * np.sin(w*(t-t_f) + theta0 + 2*np.pi/3))

        self._fig_fc.clear()
        gs  = self._fig_fc.add_gridspec(2, 1, hspace=0.45)
        ax1 = self._fig_fc.add_subplot(gs[0])
        ax2 = self._fig_fc.add_subplot(gs[1])

        ax1.plot(t * 1000, i_total, "b-",  lw=1.5, label="Phase A")
        ax1.plot(t * 1000, i_B,     "r--", lw=1.0, label="Phase B", alpha=0.7)
        ax1.plot(t * 1000, i_C,     "g:",  lw=1.0, label="Phase C", alpha=0.7)
        ax1.axvline(t_f * 1000, color="k", ls="--", lw=1.5,
                    label=f"Fault at {t_f*1000:.0f} ms")
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Current (A)")
        ax1.set_title("3-Phase Symmetrical Fault Current (with DC offset)")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Fault current envelope
        env_t = t[t >= t_f]
        env_ac = I_peak * np.ones_like(env_t)
        env_dc = abs(I_dc0) * np.exp(-(env_t - t_f) / max(tau_dc, 1e-9))
        ax2.fill_between(env_t * 1000, -(env_ac + env_dc), env_ac + env_dc,
                         alpha=0.2, color="blue", label="Asymmetric envelope")
        ax2.fill_between(env_t * 1000, -env_ac, env_ac,
                         alpha=0.3, color="orange", label="AC envelope")
        ax2.plot(t * 1000, i_total, "b-", lw=1.5)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Current (A)")
        ax2.set_title("Current Envelope Analysis")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        self._fig_fc.tight_layout()
        self._c_fc.draw_idle()

        self._fc_info.set(
            f"I_sc (rms) = {I_rms:.2f} A  |  "
            f"I_peak (AC) = {I_peak:.2f} A  |  "
            f"I_peak (asymm) = {I_peak_asym:.2f} A  |  "
            f"X/R = {XR:.2f}  |  τ_dc = {tau_dc*1000:.1f} ms"
        )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 5 – Protection Coordination
    # ─────────────────────────────────────────────────────────────────────
    def _build_tab_protection(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Protection  ")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        ctrl = ttk.Frame(tab)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=4)

        for lbl, var, lo, hi in [
            ("CTI (s):", self._p_CTI,  0.1, 1.0),
            ("TMS₁:",    self._p_TMS1, 0.05, 1.0),
            ("TMS₂:",    self._p_TMS2, 0.1, 2.0),
        ]:
            ttk.Label(ctrl, text=lbl).pack(side="left", padx=4)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient="horizontal", length=90).pack(side="left")

        ttk.Button(ctrl, text="Plot Coordination Curves",
                   command=self._plot_prot).pack(side="left", padx=10)

        canv_frame = ttk.Frame(tab)
        canv_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        canv_frame.grid_rowconfigure(0, weight=1)
        canv_frame.grid_columnconfigure(0, weight=1)

        self._fig_pr = Figure(tight_layout=True)
        self._c_pr   = _embed(self._fig_pr, canv_frame)

        self._pr_info = tk.StringVar()
        ttk.Label(tab, textvariable=self._pr_info,
                  font=("Courier New", 10)).grid(row=2, column=0, pady=4)

        self._plot_prot()

    def _plot_prot(self) -> None:
        TMS1 = float(self._p_TMS1.get())
        TMS2 = float(self._p_TMS2.get())
        CTI  = float(self._p_CTI.get())

        M = np.linspace(1.01, 25.0, 600)   # current multiples of I_pickup

        # IEC Standard Inverse (SI):  t = TMS × 0.14 / (M^0.02 − 1)
        def si(m, tms):
            return tms * 0.14 / (np.maximum(m ** 0.02 - 1.0, 1e-9))

        # IEC Very Inverse (VI): t = TMS × 13.5 / (M − 1)
        def vi(m, tms):
            return tms * 13.5 / np.maximum(m - 1.0, 1e-9)

        # IEC Extremely Inverse (EI): t = TMS × 80 / (M² − 1)
        def ei(m, tms):
            return tms * 80.0 / np.maximum(m ** 2 - 1.0, 1e-9)

        I_f = 10.0   # nominal fault multiple for coordination check
        t1_f  = si(I_f, TMS1)
        t2_f  = si(I_f, TMS2)
        margin = t2_f - t1_f

        self._fig_pr.clear()
        gs  = self._fig_pr.add_gridspec(1, 2, wspace=0.35)
        ax1 = self._fig_pr.add_subplot(gs[0])
        ax2 = self._fig_pr.add_subplot(gs[1])

        for ax in (ax1, ax2):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Current Multiple  (I / I_pickup)")
            ax.set_ylabel("Operating Time (s)")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_ylim(0.01, 100)
            ax.set_xlim(1.01, 30)

        # Standard Inverse
        ax1.plot(M, si(M, TMS1), "b-",  lw=2, label=f"Relay 1 (main)  TMS={TMS1:.2f}")
        ax1.plot(M, si(M, TMS2), "r-",  lw=2, label=f"Relay 2 (backup) TMS={TMS2:.2f}")
        ax1.axvline(I_f, color="k", ls=":", lw=1.5, label=f"Fault = {I_f}×Ip")
        ax1.annotate(
            f"Margin\n{margin:.3f} s",
            xy=(I_f, (t1_f + t2_f) / 2),
            xytext=(I_f * 1.5, (t1_f + t2_f) / 2),
            fontsize=9, arrowprops=dict(arrowstyle="->"),
        )
        ax1.set_title("IEC Standard Inverse (SI)")
        ax1.legend(fontsize=9)

        # Compare curve types for relay 1
        ax2.plot(M, si(M, TMS1), "b-",  lw=2, label="Standard Inverse")
        ax2.plot(M, vi(M, TMS1), "r--", lw=2, label="Very Inverse")
        ax2.plot(M, ei(M, TMS1), "g:",  lw=2, label="Extremely Inverse")
        ax2.set_title(f"Curve Comparison – Relay 1 (TMS={TMS1:.2f})")
        ax2.legend(fontsize=9)

        self._fig_pr.tight_layout()
        self._c_pr.draw_idle()

        ok = margin >= CTI
        self._pr_info.set(
            f"Margin @ {I_f}× pickup = {margin:.3f} s  |  "
            f"Required CTI = {CTI:.2f} s  |  "
            f"{'✓ COORDINATED' if ok else '✗ MISCOORDINATED  – increase TMS₂ or reduce TMS₁'}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 6 – Speed Controller (PID / Fuzzy / PI)
    # ─────────────────────────────────────────────────────────────────────
    def _build_tab_speed_ctrl(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Speed Controller  ")
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # ── Controller selector ──────────────────────────────────────
        row0 = ttk.Frame(tab)
        row0.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        ttk.Label(row0, text="Controller type:").pack(side="left")
        for ctype in ("PID", "PI", "Fuzzy"):
            ttk.Radiobutton(row0, text=ctype, variable=self._ctrl_type,
                            value=ctype).pack(side="left", padx=6)

        # ── Gains / setpoints ────────────────────────────────────────
        row1 = ttk.Frame(tab)
        row1.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        for lbl, var, lo, hi in [
            ("Kp:",            self.Kp,     0.1,  20.0),
            ("Ki:",            self.Ki,     0.0,   5.0),
            ("Kd:",            self.Kd,     0.0,   2.0),
            ("N_ref (rpm):",   self.N_ref,  200, 4000),
            ("Step Load (N·m):", self.T_step, 0,  150),
            ("Step Time (s):", self.t_step, 0.5,   5.0),
        ]:
            ttk.Label(row1, text=lbl).pack(side="left", padx=3)
            ttk.Scale(row1, from_=lo, to=hi, variable=var,
                      orient="horizontal", length=80).pack(side="left")

        ttk.Button(row1, text="▶ Run Simulation",
                   command=self._run_speed_sim).pack(side="left", padx=10)

        canv_frame = ttk.Frame(tab)
        canv_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        canv_frame.grid_rowconfigure(0, weight=1)
        canv_frame.grid_columnconfigure(0, weight=1)

        self._fig_sc = Figure(tight_layout=True)
        self._c_sc   = _embed(self._fig_sc, canv_frame)

        self._run_speed_sim()

    def _run_speed_sim(self) -> None:
        p       = self._p()
        Ns      = p["Ns"]; ws = p["ws"]
        V1      = p["V1"]; R1 = p["R1"]; R2 = p["R2"]
        X1      = p["X1"]; X2 = p["X2"]
        Kp      = float(self.Kp.get())
        Ki      = float(self.Ki.get())
        Kd      = float(self.Kd.get())
        N_ref   = float(self.N_ref.get())
        T_load  = float(self.T_step.get())
        t_step  = float(self.t_step.get())
        ctype   = self._ctrl_type.get()
        J       = 0.5   # kg·m²
        dt      = 0.001
        t_end   = 6.0

        t_arr = np.arange(0.0, t_end, dt)
        N     = np.empty_like(t_arr)
        Te    = np.empty_like(t_arr)
        u_arr = np.empty_like(t_arr)

        N[0]  = N_ref
        Te[0] = 0.0
        u_arr[0] = 0.0

        integ    = 0.0
        prev_err = 0.0

        # Fuzzy helper
        def _fuzzy(err: float, derr: float) -> float:
            en  = np.clip(err  / 500.0,   -1.0, 1.0)
            dn  = np.clip(derr / 5000.0,  -1.0, 1.0)
            return float(np.clip(3.0 * en + 0.8 * dn, -6.0, 6.0))

        for i in range(1, len(t_arr)):
            Tl_i = T_load if t_arr[i] >= t_step else 0.0
            err  = N_ref - N[i - 1]

            if ctype == "PID":
                integ   += err * dt
                integ    = float(np.clip(integ, -300.0, 300.0))
                derr     = (err - prev_err) / dt
                u        = Kp * err + Ki * integ + Kd * derr
            elif ctype == "PI":
                integ   += err * dt
                integ    = float(np.clip(integ, -300.0, 300.0))
                u        = Kp * err + Ki * integ
                derr     = 0.0
            else:       # Fuzzy
                derr     = (err - prev_err) / dt
                u        = _fuzzy(err, derr) * 150.0
            prev_err = err
            u_arr[i] = u

            # Effective slip: controller nudges slip around operating point
            # Positive u → increase Te (reduce effective R2/s)
            s_nom = float(np.clip((Ns - N[i - 1]) / (Ns + 1e-9), 1e-4, 0.99))
            s_eff = float(np.clip(s_nom - u * 2e-5, 1e-4, 0.99))
            denom = (R1 + R2 / s_eff) ** 2 + (X1 + X2) ** 2
            Te[i] = (3.0 / ws) * V1 ** 2 * (R2 / s_eff) / denom
            Te[i] = float(np.clip(Te[i], 0.0, 5000.0))

            domega = (Te[i] - Tl_i) / J
            N[i]   = max(N[i - 1] + domega * (60.0 / (2.0 * np.pi)) * dt, 0.0)

        self._fig_sc.clear()
        gs  = self._fig_sc.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
        ax1 = self._fig_sc.add_subplot(gs[0, :])
        ax2 = self._fig_sc.add_subplot(gs[1, 0])
        ax3 = self._fig_sc.add_subplot(gs[1, 1])

        ax1.plot(t_arr, N, "b-", lw=1.5)
        ax1.axhline(N_ref, color="red", ls="--", lw=1.0,
                    label=f"Reference {N_ref:.0f} rpm")
        ax1.axvline(t_step, color="orange", ls=":", lw=1.5,
                    label=f"Step load {T_load:.0f} N·m at {t_step:.1f}s")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Speed (rpm)")
        ax1.set_title(f"Speed Response – {ctype} Controller (Wound-Rotor / Scherbius Drive)")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_arr, Te, "g-", lw=1.5, label="T_e")
        ax2.axhline(T_load, color="r", ls="--", lw=1, label=f"Load {T_load:.0f} N·m")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Torque (N·m)")
        ax2.set_title("Electromagnetic Torque")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        ax3.plot(t_arr, u_arr, "m-", lw=1.5)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Controller Output  u")
        ax3.set_title("Control Signal")
        ax3.grid(True, alpha=0.3)

        self._fig_sc.tight_layout()
        self._c_sc.draw_idle()

    # ─────────────────────────────────────────────────────────────────────
    # TAB 7 – Thermal Analysis
    # ─────────────────────────────────────────────────────────────────────
    def _build_tab_thermal(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Thermal  ")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        ctrl = ttk.Frame(tab)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        for lbl, var, lo, hi in [
            ("Rth (°C/W):", self._Rth,  0.001, 0.1),
            ("Cth (J/°C):", self._Cth,  50,  2000),
            ("T_ambient (°C):", self._Ta, 20,  60),
            ("Load factor:", self._kl,  0.1,  1.3),
        ]:
            ttk.Label(ctrl, text=lbl).pack(side="left", padx=4)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient="horizontal", length=90).pack(side="left")
        ttk.Button(ctrl, text="Run Thermal Simulation",
                   command=self._plot_thermal).pack(side="left", padx=10)

        canv_frame = ttk.Frame(tab)
        canv_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        canv_frame.grid_rowconfigure(0, weight=1)
        canv_frame.grid_columnconfigure(0, weight=1)

        self._fig_th = Figure(tight_layout=True)
        self._c_th   = _embed(self._fig_th, canv_frame)

        self._th_info = tk.StringVar()
        ttk.Label(tab, textvariable=self._th_info,
                  font=("Courier New", 10)).grid(row=2, column=0, pady=4)

        self._plot_thermal()

    def _plot_thermal(self) -> None:
        p    = self._p()
        Rth  = float(self._Rth.get())
        Cth  = float(self._Cth.get())
        Ta   = float(self._Ta.get())
        kl   = float(self._kl.get())
        V1   = p["V1"]; R1 = p["R1"]; R2 = p["R2"]
        X1   = p["X1"]; X2 = p["X2"]; Xm = p["Xm"]

        # Operating current estimate at sub-sync operating point
        s    = p["s_sub"]
        Z_in = complex(R1 + R2 / s, X1 + X2)
        I_op = V1 / abs(Z_in)

        P_s   = 3.0 * R1 * I_op ** 2 * kl ** 2        # stator copper loss
        P_r   = 3.0 * R2 * I_op ** 2 * kl ** 2        # rotor copper loss
        P_core = 0.015 * p["VL"] ** 2 / Xm             # iron loss approx
        P_fw  = 0.005 * p["VL"] ** 2 / Xm              # friction & windage
        P_tot = P_s + P_r + P_core + P_fw

        tau_th = Rth * Cth
        T_ss   = Ta + P_tot * Rth

        t_sim  = np.linspace(0.0, 5.5 * tau_th, 1000)
        T_rise = T_ss - (T_ss - Ta) * np.exp(-t_sim / max(tau_th, 1e-9))

        # Class F insulation limit
        T_lim = 155.0

        self._fig_th.clear()
        gs  = self._fig_th.add_gridspec(1, 2, wspace=0.35)
        ax1 = self._fig_th.add_subplot(gs[0])
        ax2 = self._fig_th.add_subplot(gs[1])

        ax1.plot(t_sim / 60.0, T_rise, "r-", lw=2, label="Winding temperature")
        ax1.axhline(T_lim, color="k", ls="--", lw=1.5,
                    label=f"Class F limit {T_lim} °C")
        ax1.axhline(T_ss, color="orange", ls=":", lw=1.5,
                    label=f"T_steady = {T_ss:.1f} °C")
        if T_ss > T_lim:
            ax1.fill_between(t_sim / 60.0, T_lim, T_rise,
                             where=T_rise > T_lim,
                             alpha=0.3, color="red", label="Overtemperature")
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Temperature (°C)")
        ax1.set_title("Thermal Transient – Winding Temperature")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Loss breakdown
        labels  = ["Stator\nI²R", "Rotor\nI²R", "Core\nLoss", "Friction\n& Windage"]
        vals    = [P_s, P_r, P_core, P_fw]
        colors  = ["#4472C4", "#ED7D31", "#A9D18E", "#FFD966"]
        bars    = ax2.bar(labels, vals, color=colors, edgecolor="black", width=0.55)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2.0,
                     bar.get_height() + 0.5,
                     f"{v:.1f} W", ha="center", fontsize=9)
        ax2.set_ylabel("Power (W)")
        ax2.set_title(f"Loss Breakdown  (kl = {kl:.2f})")
        ax2.grid(True, axis="y", alpha=0.3)

        self._fig_th.tight_layout()
        self._c_th.draw_idle()

        status = "SAFE ✓" if T_ss < T_lim else "OVERTEMPERATURE ✗"
        self._th_info.set(
            f"P_total = {P_tot:.1f} W  |  τ_th = {tau_th:.0f} s "
            f"({tau_th/60:.1f} min)  |  T_steady = {T_ss:.1f} °C  |  {status}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 8 – Economic Analysis
    # ─────────────────────────────────────────────────────────────────────
    def _build_tab_economic(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Economic  ")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        ctrl = ttk.Frame(tab)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        for lbl, var, lo, hi in [
            ("Tariff ($/kWh):", self._tariff,    0.05, 0.50),
            ("Hrs/year:",       self._hours,     1000, 8760),
            ("Rated (kW):",     self._Prated_kW, 1,    200),
        ]:
            ttk.Label(ctrl, text=lbl).pack(side="left", padx=4)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient="horizontal", length=90).pack(side="left")
        ttk.Button(ctrl, text="Calculate Economics",
                   command=self._plot_econ).pack(side="left", padx=10)

        canv_frame = ttk.Frame(tab)
        canv_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        canv_frame.grid_rowconfigure(0, weight=1)
        canv_frame.grid_columnconfigure(0, weight=1)

        self._fig_ec = Figure(tight_layout=True)
        self._c_ec   = _embed(self._fig_ec, canv_frame)

        self._ec_info = tk.StringVar()
        ttk.Label(tab, textvariable=self._ec_info,
                  font=("Courier New", 10)).grid(row=2, column=0, pady=4)

        self._plot_econ()

    def _plot_econ(self) -> None:
        tariff = float(self._tariff.get())
        hours  = float(self._hours.get())
        Pr_W   = float(self._Prated_kW.get()) * 1000.0
        p      = self._p()
        s_sub  = p["s_sub"]

        # Standard motor efficiency approximation
        eff_std = 0.92

        # Scherbius: slip energy recovered → effective efficiency improves
        eff_sch = min(0.97, eff_std + s_sub * (1.0 - eff_std) * 0.80)

        lf = np.linspace(0.2, 1.2, 60)
        P_out = Pr_W * lf
        P_in_std = P_out / eff_std
        P_in_sch = P_out / eff_sch

        E_std  = P_in_std * hours / 1000.0
        E_sch  = P_in_sch * hours / 1000.0
        cost_std = E_std * tariff
        cost_sch = E_sch * tariff
        savings_arr = cost_std - cost_sch

        # Investment cost of Scherbius converter ≈ 30 % of motor cost
        motor_cost = Pr_W / 1000.0 * 500.0      # $500/kW
        inv_cost   = motor_cost * 0.30

        # Payback at full load (index for lf ≈ 1.0)
        idx_fl     = np.argmin(np.abs(lf - 1.0))
        save_fl    = float(savings_arr[idx_fl])
        payback_yr = inv_cost / save_fl if save_fl > 0 else float("inf")

        # 10-year NPV at 8 % discount
        discount = 0.08
        npv = -inv_cost + sum(save_fl / (1.0 + discount) ** yr
                              for yr in range(1, 11))

        self._fig_ec.clear()
        gs  = self._fig_ec.add_gridspec(1, 2, wspace=0.35)
        ax1 = self._fig_ec.add_subplot(gs[0])
        ax2 = self._fig_ec.add_subplot(gs[1])

        ax1.plot(lf * 100, cost_std / 1000, "r-",  lw=2, label="Standard motor")
        ax1.plot(lf * 100, cost_sch / 1000, "g-",  lw=2, label="With Scherbius")
        ax1.fill_between(lf * 100, cost_sch / 1000, cost_std / 1000,
                         alpha=0.2, color="green", label="Annual savings")
        ax1.set_xlabel("Load Factor (%)")
        ax1.set_ylabel("Annual Energy Cost  (k$)")
        ax1.set_title("Energy Cost Comparison")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        years   = np.arange(1, 11)
        cum_sav = save_fl * years
        ax2.bar(years, cum_sav / 1000, color="steelblue", edgecolor="black",
                width=0.6)
        ax2.axhline(inv_cost / 1000, color="r", ls="--", lw=2,
                    label=f"Investment  ${inv_cost/1000:.1f}k")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Cumulative Savings  (k$)")
        ax2.set_title(f"Payback Analysis  (NPV 10yr = ${npv/1000:.1f}k)")
        ax2.legend(fontsize=9)
        ax2.grid(True, axis="y", alpha=0.3)

        self._fig_ec.tight_layout()
        self._c_ec.draw_idle()

        self._ec_info.set(
            f"η_std = {eff_std*100:.1f}%  →  η_Scherbius = {eff_sch*100:.1f}%  |  "
            f"Annual saving @ FL = ${save_fl:.0f}  |  "
            f"Investment = ${inv_cost:.0f}  |  "
            f"Simple payback = {payback_yr:.1f} yr  |  NPV = ${npv:.0f}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 9 – Harmonic & Power Quality
    # ─────────────────────────────────────────────────────────────────────
    def _build_tab_harmonic(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Harmonics  ")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        ctrl = ttk.Frame(tab)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        ttk.Label(ctrl, text="Max harmonic order:").pack(side="left", padx=4)
        ttk.Scale(ctrl, from_=3, to=25, variable=self._h_order,
                  orient="horizontal", length=100).pack(side="left")
        ttk.Label(ctrl, text="Disp. PF:").pack(side="left", padx=8)
        ttk.Scale(ctrl, from_=0.5, to=1.0, variable=self._h_dpf,
                  orient="horizontal", length=90).pack(side="left")
        ttk.Button(ctrl, text="Analyse Harmonics",
                   command=self._plot_harm).pack(side="left", padx=10)

        canv_frame = ttk.Frame(tab)
        canv_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        canv_frame.grid_rowconfigure(0, weight=1)
        canv_frame.grid_columnconfigure(0, weight=1)

        self._fig_hm = Figure(tight_layout=True)
        self._c_hm   = _embed(self._fig_hm, canv_frame)

        self._hm_info = tk.StringVar()
        ttk.Label(tab, textvariable=self._hm_info,
                  font=("Courier New", 10)).grid(row=2, column=0, pady=4)

        self._plot_harm()

    def _plot_harm(self) -> None:
        p       = self._p()
        f_s     = p["f_s"]
        V1      = p["V1"]
        R1, X1  = p["R1"], p["X1"]
        R2, X2  = p["R2"], p["X2"]
        max_ord = int(round(float(self._h_order.get())))
        dpf     = float(self._h_dpf.get())

        # Fundamental current
        s   = p["s_sub"]
        I1  = V1 / abs(complex(R1 + R2 / s, X1 + X2))

        # Typical 6-pulse converter harmonic current spectrum (IEC 61000-3-2)
        h_all = [1, 5, 7, 11, 13, 17, 19, 23, 25]
        h_pct = {1: 100.0, 5: 20.0, 7: 14.0, 11: 9.0,
                 13: 7.0, 17: 5.8, 19: 5.1, 23: 4.3, 25: 4.0}

        orders = [h for h in h_all if h <= max_ord]
        mags   = [h_pct.get(h, 0.0) * I1 / 100.0 for h in orders]

        THD_I = np.sqrt(sum(
            (h_pct.get(h, 0.0) / 100.0) ** 2 for h in orders if h > 1
        )) * 100.0

        PF = dpf / np.sqrt(1.0 + (THD_I / 100.0) ** 2)

        # Reconstructed waveform (2 cycles)
        t = np.linspace(0.0, 2.0 / f_s, 2500)
        i_wave = np.zeros_like(t)
        for h, mag in zip(orders, mags):
            i_wave += mag * np.sqrt(2.0) * np.sin(2.0 * np.pi * h * f_s * t)

        # Voltage THD estimate (simplified: each harmonic voltage ≈ Ih × Z_h)
        V_h_sq = 0.0
        for h, mag in zip(orders, mags):
            if h > 1:
                Z_h = abs(complex(R1, h * (X1 + X2)))
                V_h_sq += (mag * Z_h) ** 2
        THD_V = np.sqrt(V_h_sq) / V1 * 100.0

        self._fig_hm.clear()
        gs  = self._fig_hm.add_gridspec(1, 2, wspace=0.35)
        ax1 = self._fig_hm.add_subplot(gs[0])
        ax2 = self._fig_hm.add_subplot(gs[1])

        ax1.plot(t * 1000, i_wave, "b-", lw=1.5)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Current (A)")
        ax1.set_title("Supply Current Waveform (with harmonics)")
        ax1.grid(True, alpha=0.3)

        colors = ["green" if h == 1 else "orange" if h <= 7 else "red"
                  for h in orders]
        bars = ax2.bar([str(h) for h in orders], mags,
                       color=colors, edgecolor="black")
        for bar, m in zip(bars, mags):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.05,
                f"{m:.2f}",
                ha="center",
                fontsize=8,
            )
        ax2.set_xlabel("Harmonic Order")
        ax2.set_ylabel("Current Magnitude  (A)")
        ax2.set_title(f"Harmonic Spectrum  (THD_I = {THD_I:.1f} %)")
        ax2.grid(True, axis="y", alpha=0.3)

        # IEEE 519 limit bar
        ax2.axhline(I1 * 0.08, color="red", ls="--", lw=1.5,
                    label="IEEE 519 (8 % I₁)")
        ax2.legend(fontsize=9)

        self._fig_hm.tight_layout()
        self._c_hm.draw_idle()

        ieee_ok = THD_I <= 5.0
        self._hm_info.set(
            f"THD_I = {THD_I:.1f}%  |  THD_V ≈ {THD_V:.1f}%  |  "
            f"DPF = {dpf:.3f}  |  True PF = {PF:.3f}  |  "
            f"I₁ = {I1:.2f} A  |  "
            f"IEEE 519 (5%): {'PASS ✓' if ieee_ok else 'FAIL ✗'}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # TAB 10 – Comprehensive Analysis Dashboard
    # ─────────────────────────────────────────────────────────────────────
    def _build_tab_comprehensive(self) -> None:
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="  Comprehensive  ")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        ctrl = ttk.Frame(tab)
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        ttk.Button(ctrl, text="Generate Full Dashboard",
                   command=self._plot_comprehensive).pack(side="left", padx=6)
        ttk.Button(ctrl, text="Export Text Report",
                   command=self._export_report).pack(side="left", padx=6)

        canv_frame = ttk.Frame(tab)
        canv_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        canv_frame.grid_rowconfigure(0, weight=1)
        canv_frame.grid_columnconfigure(0, weight=1)

        self._fig_co = Figure(tight_layout=True)
        self._c_co   = _embed(self._fig_co, canv_frame)

        self._plot_comprehensive()

    def _plot_comprehensive(self) -> None:
        p   = self._p()
        Ns  = p["Ns"]; ws = p["ws"]
        V1  = p["V1"]; R1 = p["R1"]; R2 = p["R2"]
        X1  = p["X1"]; X2 = p["X2"]

        s_arr = np.linspace(1e-3, 1.0, 500)
        N_arr = Ns * (1.0 - s_arr)

        def _Te(s):
            d = (R1 + R2 / s) ** 2 + (X1 + X2) ** 2
            return (3.0 / ws) * V1 ** 2 * (R2 / s) / d

        Te_arr   = np.array([_Te(s) for s in s_arr])
        Pag_arr  = Te_arr * ws
        Pmech_arr = (1.0 - s_arr) * Pag_arr
        Pslip_arr = s_arr * Pag_arr

        # Input power & efficiency (simplified)
        I_arr   = V1 / np.sqrt((R1 + R2 / s_arr) ** 2 + (X1 + X2) ** 2)
        Pin_arr = Pag_arr + 3.0 * R1 * I_arr ** 2
        eff_arr = np.where(Pin_arr > 0, Pmech_arr / Pin_arr, 0.0)
        eff_arr = np.clip(eff_arr, 0.0, 1.0)

        self._fig_co.clear()
        gs = self._fig_co.add_gridspec(2, 3, hspace=0.5, wspace=0.38)

        # ── 1. Torque-speed ───────────────────────────────────────────
        ax1 = self._fig_co.add_subplot(gs[0, 0])
        ax1.plot(N_arr, Te_arr, "b-", lw=2)
        for N_op, col in [(p["N_sub"], "green"), (p["N_sup"], "purple")]:
            ax1.axvline(N_op, color=col, ls="--", lw=1.5)
        ax1.set_xlabel("Speed (rpm)")
        ax1.set_ylabel("Torque (N·m)")
        ax1.set_title("Torque-Speed")
        ax1.grid(True, alpha=0.3)

        # ── 2. Power flow ─────────────────────────────────────────────
        ax2 = self._fig_co.add_subplot(gs[0, 1])
        ax2.plot(N_arr, Pag_arr  / 1000, "r-",  lw=2, label="Air-gap")
        ax2.plot(N_arr, Pmech_arr / 1000, "g-",  lw=2, label="Mechanical")
        ax2.plot(N_arr, Pslip_arr / 1000, "b--", lw=2, label="Slip power")
        ax2.set_xlabel("Speed (rpm)")
        ax2.set_ylabel("Power (kW)")
        ax2.set_title("Power Flow")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # ── 3. Efficiency ─────────────────────────────────────────────
        ax3 = self._fig_co.add_subplot(gs[0, 2])
        ax3.plot(N_arr, eff_arr * 100, "m-", lw=2)
        ax3.axvline(p["N_sub"], color="green", ls="--", lw=1.5,
                    label=f"Sub-sync {p['N_sub']:.0f} rpm")
        ax3.set_xlabel("Speed (rpm)")
        ax3.set_ylabel("Efficiency (%)")
        ax3.set_title("Efficiency Curve")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # ── 4. Phasor diagram at sub-sync operating point ─────────────
        ax4 = self._fig_co.add_subplot(gs[1, 0], projection="polar")
        s_op  = p["s_sub"]
        Z_op  = complex(R1 + R2 / s_op, X1 + X2)
        I_mag = V1 / abs(Z_op)
        phi   = np.angle(Z_op)

        ax4.annotate("", xy=(0.0, V1), xytext=(0.0, 0.0),
                     arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax4.annotate("", xy=(-phi, I_mag), xytext=(0.0, 0.0),
                     arrowprops=dict(arrowstyle="->", color="blue", lw=2))
        ax4.set_title(f"Phasor (s={s_op:.2f})\nV₁={V1:.0f}V, I={I_mag:.1f}A",
                      pad=20, fontsize=9)

        # ── 5. Parameter summary table ────────────────────────────────
        ax5 = self._fig_co.add_subplot(gs[1, 1])
        ax5.axis("off")
        rows = [
            ["Parameter",              "Value",              "Unit"],
            ["Stator freq f_s",        f"{p['f_s']:.0f}",    "Hz"],
            ["Rotor inj. f_r",         f"{p['f_r']:.0f}",    "Hz"],
            ["Poles",                  f"{p['pol']}",         ""],
            ["Sync speed Ns",          f"{p['Ns']:.0f}",     "rpm"],
            ["Sub-sync slip s_sub",    f"{p['s_sub']:.3f}",  "pu"],
            ["Speed N₁ (lower)",       f"{p['N_sub']:.0f}",  "rpm"],
            ["Super-sync slip s_sup",  f"{p['s_sup']:.3f}",  "pu"],
            ["Speed N₂ (higher)",      f"{p['N_sup']:.0f}",  "rpm"],
            ["Torque @ N₁",            f"{p['T_sub']:.1f}",  "N·m"],
            ["Slip power @ N₁",        f"{p['P_slip']:.0f}", "W (absorbed)"],
        ]
        tbl = ax5.table(
            cellText=[r[1:] for r in rows[1:]],
            rowLabels=[r[0] for r in rows[1:]],
            colLabels=["Value", "Unit"],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.35)
        ax5.set_title("Key Parameters", fontsize=10)

        # ── 6. Power distribution pie at sub-sync ─────────────────────
        ax6 = self._fig_co.add_subplot(gs[1, 2])
        Pm_sub   = float(np.clip(p["P_mech_sub"], 0, None))
        Psl_sub  = float(np.clip(p["P_slip"],     0, None))
        I_s      = V1 / abs(complex(R1 + R2 / p["s_sub"], X1 + X2))
        Pcu_s    = 3.0 * R1 * I_s ** 2
        pieces   = [Pm_sub, Psl_sub, Pcu_s]
        labels_p = ["Mech. Output", "Slip Recovery", "Stator I²R"]
        colors_p = ["#4CAF50", "#2196F3", "#F44336"]
        non_zero = [(v, l, c) for v, l, c in zip(pieces, labels_p, colors_p)
                    if v > 0]
        if non_zero:
            vals_nz, lbl_nz, col_nz = zip(*non_zero)
            ax6.pie(vals_nz, labels=lbl_nz, colors=col_nz,
                    autopct="%1.1f%%", startangle=90)
        ax6.set_title(
            f"Power Distribution\n@ {p['N_sub']:.0f} rpm  (s = {p['s_sub']:.2f})",
            fontsize=9,
        )

        self._fig_co.tight_layout()
        self._c_co.draw_idle()

    def _export_report(self) -> None:
        p = self._p()
        msg = (
            "═══ Wound-Rotor DFIM – Engineering Report ═══\n\n"
            f"SUPPLY:  VL={p['VL']:.0f} V, f_s={p['f_s']:.0f} Hz, "
            f"poles={p['pol']}\n"
            f"ROTOR INJECTION:  f_r={p['f_r']:.0f} Hz\n\n"
            f"SYNCHRONOUS SPEED:  Ns = {p['Ns']:.0f} rpm\n\n"
            "── SUBSYNCHRONOUS ──\n"
            f"  Slip       s_sub = {p['s_sub']:.4f}\n"
            f"  Speed      N₁   = {p['N_sub']:.1f} rpm\n"
            f"  Torque     T₁   = {p['T_sub']:.2f} N·m\n"
            f"  Air-gap P       = {p['P_ag_sub']:.1f} W\n"
            f"  Slip power(abs) = {p['P_slip']:.1f} W  ← 11 Hz source ABSORBS\n\n"
            "── SUPERSYNCHRONOUS ──\n"
            f"  Slip       s_sup = {p['s_sup']:.4f}\n"
            f"  Speed      N₂   = {p['N_sup']:.1f} rpm\n"
            f"  Torque     T₂   = {p['T_sup']:.2f} N·m\n"
            f"  Slip power(del) = {p['P_slip_sup']:.1f} W  ← 11 Hz source DELIVERS\n\n"
            "ANSWER SUMMARY:\n"
            "  The rotor can turn at 2340 rpm or 3660 rpm.\n"
            "  At the LOWER speed (2340 rpm) the 11 Hz source\n"
            "  ABSORBS active power (Scherbius / slip-power recovery).\n"
        )
        messagebox.showinfo("Engineering Report", msg)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    root = tk.Tk()
    app = DFIMApp(root)  # noqa: F841
    root.mainloop()


if __name__ == "__main__":
    main()
