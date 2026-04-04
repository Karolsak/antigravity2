"""
DC Motor Advanced Interactive Engineering Laboratory (Tkinter)
==============================================================
Comprehensive simulation suite for DC motors:
  - Separately Excited, Shunt, Series, Compound (Cumulative/Differential)
  - Electromechanical ODE simulation (RK4, dt=1ms)
  - Thermal model, PID controller, protection, disturbances, scenarios
  - 10 interactive notebook tabs

Usage:
    python dc_motor_suite.py
"""

from __future__ import annotations

import csv
import math
import os
import time
from collections import deque

import matplotlib
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOTOR_TYPES = [
    "Separately Excited",
    "Shunt",
    "Series",
    "Compound Cumulative",
    "Compound Differential",
]
LOAD_TYPES = ["Constant", "Quadratic (Fan/Pump)", "Linear (Friction)"]
INSULATION_CLASS = {"A": 105, "B": 130, "F": 155, "H": 180}
MAX_HISTORY = 2000
DT = 0.001          # physics step [s]
STEPS_PER_FRAME = 20


# ---------------------------------------------------------------------------
# Motor physics model
# ---------------------------------------------------------------------------

class DCMotorModel:
    """
    Electro-thermal DC motor model.

    State: [Ia, omega, T_arm, T_fld]

    Supports four winding configurations; integrates with 4th-order
    Runge-Kutta.
    """

    def __init__(self) -> None:
        # Electrical
        self.Ra0: float = 0.5       # armature resistance at 25°C [Ω]
        self.La: float = 0.020      # armature inductance [H]
        self.Rf: float = 200.0      # field winding resistance [Ω]
        self.Lf: float = 5.0        # field inductance [H] (not used in steady-state)
        self.Rf_s: float = 0.15     # series field resistance [Ω]
        self.Ke: float = 0.5        # back-EMF constant [V·s/rad]
        self.Kt: float = 0.5        # torque constant [N·m/A]
        # Saturation (series motor Froelich curve)
        self.phi_max: float = 1.0   # normalised max flux
        self.Ia_sat: float = 15.0   # saturation current [A]
        # Mechanical
        self.J: float = 0.05        # moment of inertia [kg·m²]
        self.B: float = 0.002       # viscous friction [N·m·s/rad]
        # Supply
        self.Va: float = 220.0      # armature voltage [V]
        self.Vf: float = 220.0      # field voltage [V]
        # Load
        self.TL_rated: float = 10.0 # rated load torque [N·m]
        self.load_type: str = "Constant"
        self.motor_type: str = "Separately Excited"
        # Thermal
        self.T_amb: float = 25.0    # ambient temperature [°C]
        self.Rth_a: float = 2.0     # armature thermal resistance [°C/W]
        self.Cth_a: float = 500.0   # armature thermal capacitance [J/°C]
        self.Rth_f: float = 5.0     # field thermal resistance [°C/W]
        self.Cth_f: float = 200.0   # field thermal capacitance [J/°C]
        self.ins_class: str = "F"   # insulation class
        # Protection
        self.I_trip: float = 50.0   # overcurrent trip [A]
        self.omega_max: float = 350.0  # overspeed trip [rad/s]
        self.V_min: float = 80.0    # under-voltage trip [V]
        self.i2t_limit: float = 10000.0  # I²t trip [A²·s]
        # Disturbance state (transient overrides)
        self.Va_dist: float = 1.0   # multiplier on Va
        self.TL_dist: float = 1.0   # multiplier on TL
        self.Vf_dist: float = 1.0   # multiplier on Vf
        self.R_start: float = 0.0   # extra series resistance (soft-start)
        self.braking: bool = False   # dynamic braking active
        # Starting resistance steps
        self.R_start_val: float = 0.0
        # State
        self.Ia: float = 0.0
        self.omega: float = 0.0
        self.T_arm: float = 25.0
        self.T_fld: float = 25.0
        self.t: float = 0.0
        # Accumulated I²t
        self.i2t_acc: float = 0.0
        # PID
        self.pid_enabled: bool = False
        self.pid_setpoint: float = 157.0  # rad/s ≈ 1500 RPM
        self.Kp: float = 5.0
        self.Ki: float = 2.0
        self.Kd: float = 0.05
        self._pid_integral: float = 0.0
        self._pid_prev_err: float = 0.0
        self.pid_va_min: float = 0.0
        self.pid_va_max: float = 440.0
        # Trip flag
        self.tripped: bool = False
        self.trip_log: list[str] = []

    def reset(self) -> None:
        self.Ia = 0.0
        self.omega = 0.0
        self.T_arm = self.T_amb
        self.T_fld = self.T_amb
        self.t = 0.0
        self.i2t_acc = 0.0
        self._pid_integral = 0.0
        self._pid_prev_err = 0.0
        self.tripped = False
        self.Va_dist = 1.0
        self.TL_dist = 1.0
        self.Vf_dist = 1.0
        self.R_start_val = 0.0
        self.braking = False

    # ------------------------------------------------------------------
    # Flux / field current
    # ------------------------------------------------------------------

    def flux_phi(self, Ia: float) -> float:
        """Normalised flux depending on motor type."""
        mt = self.motor_type
        if mt == "Series":
            # Froelich saturation curve
            Ia_abs = abs(Ia)
            return self.phi_max * Ia_abs / (Ia_abs + self.Ia_sat)
        elif mt in ("Compound Cumulative", "Compound Differential"):
            # shunt component + series component
            If_shunt = self.Vf_eff() / self.Rf
            phi_shunt = If_shunt * self.Ke  # V·s/rad based
            # normalise by rated
            phi_norm_shunt = phi_shunt / (self.Ke * self.Vf / self.Rf)
            Ia_abs = abs(Ia)
            phi_series = 0.3 * Ia_abs / (Ia_abs + self.Ia_sat)  # 30% contribution
            if mt == "Compound Cumulative":
                return min(1.5, phi_norm_shunt + phi_series)
            else:
                return max(0.05, phi_norm_shunt - phi_series)
        else:
            # Separately excited or shunt: flux from field voltage
            If = self.Vf_eff() / self.Rf
            phi_norm = If * self.Ke / (self.Ke * self.Vf / self.Rf)
            return max(0.01, phi_norm)

    def Vf_eff(self) -> float:
        mt = self.motor_type
        if mt == "Shunt":
            return self.Va * self.Va_dist * self.Vf_dist
        return self.Vf * self.Vf_dist

    def Va_eff(self) -> float:
        Va = self.Va * self.Va_dist
        if self.braking:
            return 0.0
        if self.pid_enabled:
            return Va  # PID overrides separately
        return Va

    def Ra_eff(self, T: float) -> float:
        """Temperature-dependent armature resistance."""
        return self.Ra0 * (1.0 + 0.00393 * (T - 25.0))

    def load_torque(self, omega: float) -> float:
        TL = self.TL_rated * self.TL_dist
        lt = self.load_type
        if lt == "Quadratic (Fan/Pump)":
            omega_rated = 157.0  # ~1500 RPM reference
            return TL * (omega / omega_rated) ** 2
        elif lt == "Linear (Friction)":
            omega_rated = 157.0
            return TL * abs(omega) / omega_rated
        return TL  # constant

    # ------------------------------------------------------------------
    # Derivatives (for RK4)
    # ------------------------------------------------------------------

    def derivatives(self, state: list[float], Va_cmd: float) -> list[float]:
        Ia, omega, T_arm, T_fld = state
        Ra = self.Ra_eff(T_arm)
        phi = self.flux_phi(Ia)

        # Effective armature voltage (considering series motor topology)
        mt = self.motor_type
        if mt == "Series":
            # series field resistance in armature loop
            R_total = Ra + self.Rf_s + self.R_start_val
            Va_loop = Va_cmd
        else:
            R_total = Ra + self.R_start_val
            Va_loop = Va_cmd

        Eb = self.Ke * phi * omega  # back EMF

        dIa = (Va_loop - R_total * Ia - Eb) / self.La

        Te = self.Kt * phi * Ia
        TL = self.load_torque(omega)

        # For series motor keep omega >= 0 (no reversal)
        if mt == "Series" and omega <= 0 and (Te - TL) <= 0:
            dOmega = 0.0
        else:
            dOmega = (Te - self.B * omega - TL) / self.J

        # Thermal
        P_arm = Ra * Ia ** 2
        P_fld = self.Vf_eff() ** 2 / max(self.Rf, 1.0)
        if mt == "Series":
            P_fld = self.Rf_s * Ia ** 2

        dT_arm = (P_arm - (T_arm - self.T_amb) / self.Rth_a) / self.Cth_a
        dT_fld = (P_fld - (T_fld - self.T_amb) / self.Rth_f) / self.Cth_f

        return [dIa, dOmega, dT_arm, dT_fld]

    # ------------------------------------------------------------------
    # RK4 step
    # ------------------------------------------------------------------

    def step(self) -> None:
        if self.tripped:
            return

        # PID control
        if self.pid_enabled:
            err = self.pid_setpoint - self.omega
            self._pid_integral += err * DT
            # anti-windup
            self._pid_integral = float(np.clip(
                self._pid_integral, -200.0 / max(self.Ki, 1e-9), 200.0 / max(self.Ki, 1e-9)
            ))
            d_err = (err - self._pid_prev_err) / DT
            self._pid_prev_err = err
            Va_cmd = self.Kp * err + self.Ki * self._pid_integral + self.Kd * d_err
            Va_cmd = float(np.clip(Va_cmd, self.pid_va_min, self.pid_va_max))
        else:
            Va_cmd = self.Va_eff()

        state = [self.Ia, self.omega, self.T_arm, self.T_fld]
        k1 = self.derivatives(state, Va_cmd)
        s2 = [state[i] + 0.5 * DT * k1[i] for i in range(4)]
        k2 = self.derivatives(s2, Va_cmd)
        s3 = [state[i] + 0.5 * DT * k2[i] for i in range(4)]
        k3 = self.derivatives(s3, Va_cmd)
        s4 = [state[i] + DT * k3[i] for i in range(4)]
        k4 = self.derivatives(s4, Va_cmd)

        new_state = [state[i] + (DT / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
                     for i in range(4)]

        # Clamp omega to zero for series motor
        if self.motor_type == "Series":
            new_state[1] = max(0.0, new_state[1])

        self.Ia, self.omega, self.T_arm, self.T_fld = new_state
        self.t += DT

        # Accumulate I²t
        self.i2t_acc += self.Ia ** 2 * DT

        # Protection checks
        self._check_protection()

    def _check_protection(self) -> None:
        ts = f"t={self.t:.3f}s"
        if abs(self.Ia) > self.I_trip:
            self.tripped = True
            self.trip_log.append(f"[{ts}] OVERCURRENT TRIP: Ia={self.Ia:.1f} A > {self.I_trip:.1f} A")
        if self.omega > self.omega_max:
            self.tripped = True
            self.trip_log.append(f"[{ts}] OVERSPEED TRIP: ω={self.omega:.1f} rad/s > {self.omega_max:.1f}")
        if self.i2t_acc > self.i2t_limit:
            self.tripped = True
            self.trip_log.append(f"[{ts}] I²t THERMAL TRIP: {self.i2t_acc:.0f} A²s > {self.i2t_limit:.0f}")
        T_max = INSULATION_CLASS.get(self.ins_class, 155)
        if self.T_arm > T_max:
            self.tripped = True
            self.trip_log.append(f"[{ts}] INSULATION TRIP: T_arm={self.T_arm:.1f}°C > {T_max}°C")

    # ------------------------------------------------------------------
    # Steady-state calculations
    # ------------------------------------------------------------------

    def steady_state(self):
        """Return dict with steady-state operating point."""
        phi = self.flux_phi(self.TL_rated / max(self.Kt, 1e-9))  # approx
        Va = self.Va * self.Va_dist
        TL = self.TL_rated
        # Ia_rated from torque: Te = Kt*phi*Ia => Ia = TL/(Kt*phi)
        Ia_rated = TL / max(self.Kt * phi, 1e-9)
        Eb = Va - self.Ra0 * Ia_rated
        omega_rated = Eb / max(self.Ke * phi, 1e-9)
        P_in = Va * Ia_rated
        P_out = TL * omega_rated
        eta = P_out / max(P_in, 1e-9) * 100.0
        return {
            "Ia_rated": Ia_rated,
            "omega_rated": max(omega_rated, 0.0),
            "RPM_rated": max(omega_rated, 0.0) * 60 / (2 * math.pi),
            "P_in": P_in,
            "P_out": P_out,
            "eta": eta,
            "Eb": max(Eb, 0.0),
            "phi": phi,
        }


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

class DCMotorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DC Motor Advanced Engineering Laboratory")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)

        self.model = DCMotorModel()
        self.running = False
        self._after_id = None

        # History buffers
        self._hist_t     = deque(maxlen=MAX_HISTORY)
        self._hist_omega = deque(maxlen=MAX_HISTORY)
        self._hist_Ia    = deque(maxlen=MAX_HISTORY)
        self._hist_Te    = deque(maxlen=MAX_HISTORY)
        self._hist_Tarm  = deque(maxlen=MAX_HISTORY)
        self._hist_Va    = deque(maxlen=MAX_HISTORY)

        # Build UI
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=4, pady=4)

        self._build_tab1_config()
        self._build_tab2_simulation()
        self._build_tab3_tn_curves()
        self._build_tab4_starting()
        self._build_tab5_pid()
        self._build_tab6_disturbances()
        self._build_tab7_thermal()
        self._build_tab8_protection()
        self._build_tab9_power_quality()
        self._build_tab10_results()

    # ================================================================
    # Helpers
    # ================================================================

    def _make_slider(self, parent, label, var: tk.DoubleVar,
                     lo, hi, cmd=None, fmt=".3g", width=30) -> tk.Label:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=width, anchor="w").pack(side="left")
        sl = ttk.Scale(row, variable=var, from_=lo, to=hi,
                       command=lambda _: (cmd() if cmd else None) or self._update_val_label(var, val_lbl, fmt))
        sl.pack(side="left", fill="x", expand=True, padx=4)
        val_lbl = ttk.Label(row, width=9, anchor="e")
        val_lbl.pack(side="right")
        self._update_val_label(var, val_lbl, fmt)
        var.trace_add("write", lambda *a: self._update_val_label(var, val_lbl, fmt))
        return val_lbl

    @staticmethod
    def _update_val_label(var, lbl, fmt=".3g"):
        try:
            lbl.configure(text=format(var.get(), fmt))
        except Exception:
            pass

    def _make_chart(self, parent, title="", nrows=1, ncols=1,
                    figsize=(7, 4)) -> tuple:
        fig = Figure(figsize=figsize, dpi=95, tight_layout=True)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        axes = [fig.add_subplot(nrows, ncols, i+1) for i in range(nrows*ncols)]
        for ax in axes:
            ax.grid(True, alpha=0.3)
        if title and len(axes) == 1:
            axes[0].set_title(title)
        return fig, axes, canvas

    # ================================================================
    # TAB 1 – Motor Configuration
    # ================================================================

    def _build_tab1_config(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="⚙ Konfiguracja")

        left = ttk.LabelFrame(tab, text="Parametry silnika")
        left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        right = ttk.LabelFrame(tab, text="Punkt pracy (obliczony)")
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        # Motor type
        row = ttk.Frame(left)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Typ silnika:", width=30, anchor="w").pack(side="left")
        self._motor_type_var = tk.StringVar(value="Separately Excited")
        cb = ttk.Combobox(row, textvariable=self._motor_type_var, values=MOTOR_TYPES,
                          state="readonly", width=28)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", self._on_motor_type_change)

        # Load type
        row2 = ttk.Frame(left)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="Typ obciążenia:", width=30, anchor="w").pack(side="left")
        self._load_type_var = tk.StringVar(value="Constant")
        cb2 = ttk.Combobox(row2, textvariable=self._load_type_var, values=LOAD_TYPES,
                            state="readonly", width=28)
        cb2.pack(side="left")
        cb2.bind("<<ComboboxSelected>>", lambda e: self._apply_config())

        # Parameter sliders
        self._Ra_var   = tk.DoubleVar(value=0.5)
        self._La_var   = tk.DoubleVar(value=20.0)   # stored in mH
        self._Rf_var   = tk.DoubleVar(value=200.0)
        self._Ke_var   = tk.DoubleVar(value=0.5)
        self._Kt_var   = tk.DoubleVar(value=0.5)
        self._J_var    = tk.DoubleVar(value=0.05)
        self._B_var    = tk.DoubleVar(value=0.002)
        self._Va_var   = tk.DoubleVar(value=220.0)
        self._Vf_var   = tk.DoubleVar(value=220.0)
        self._TL_var   = tk.DoubleVar(value=10.0)
        self._Tamb_var = tk.DoubleVar(value=25.0)
        self._Rtha_var = tk.DoubleVar(value=2.0)
        self._Ctha_var = tk.DoubleVar(value=500.0)

        params = [
            ("Ra – opór twornika [Ω]",     self._Ra_var,   0.1,  5.0),
            ("La – indukcyjność twornika [mH]", self._La_var,  1.0, 100.0),
            ("Rf – opór wzbudzenia [Ω]",   self._Rf_var,  50.0, 500.0),
            ("Ke – stała SEM [V·s/rad]",   self._Ke_var,  0.1,  2.0),
            ("Kt – stała momentu [N·m/A]", self._Kt_var,  0.1,  2.0),
            ("J – moment bezwładności [kg·m²]", self._J_var, 0.01, 0.5),
            ("B – tłumienie lepkie [N·m·s/rad]", self._B_var, 0.0, 0.05),
            ("Va – napięcie twornika [V]", self._Va_var,  50.0, 440.0),
            ("Vf – napięcie wzbudzenia [V]", self._Vf_var, 50.0, 440.0),
            ("TL – moment znamionowy [N·m]", self._TL_var,  1.0, 100.0),
            ("T_amb – temperatura otoczenia [°C]", self._Tamb_var, 0.0, 50.0),
            ("Rth_a – res. termiczna twornika", self._Rtha_var, 0.5, 10.0),
            ("Cth_a – poj. termiczna twornika",  self._Ctha_var, 100.0, 2000.0),
        ]
        for lbl, var, lo, hi in params:
            self._make_slider(left, lbl, var, lo, hi, cmd=self._apply_config, width=35)

        # Insulation class
        row3 = ttk.Frame(left)
        row3.pack(fill="x", pady=4)
        ttk.Label(row3, text="Klasa izolacji:", width=30, anchor="w").pack(side="left")
        self._ins_var = tk.StringVar(value="F")
        ttk.Combobox(row3, textvariable=self._ins_var,
                     values=list(INSULATION_CLASS.keys()),
                     state="readonly", width=8).pack(side="left")

        ttk.Button(left, text="Zastosuj / Zresetuj symulację",
                   command=self._apply_config_and_reset).pack(pady=8)

        # Right: operating point text
        self._op_text = tk.Text(right, wrap="word", height=30, font=("Courier", 10))
        self._op_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._apply_config()

    def _on_motor_type_change(self, event=None):
        mt = self._motor_type_var.get()
        # Load sensible defaults for series motor
        if mt == "Series":
            self._Ra_var.set(0.4)
            self._Rf_var.set(80.0)
        else:
            self._Ra_var.set(0.5)
            self._Rf_var.set(200.0)
        self._apply_config()

    def _apply_config(self):
        m = self.model
        m.Ra0 = self._Ra_var.get()
        m.La  = self._La_var.get() * 0.001   # mH → H
        m.Rf  = self._Rf_var.get()
        m.Ke  = self._Ke_var.get()
        m.Kt  = self._Kt_var.get()
        m.J   = self._J_var.get()
        m.B   = self._B_var.get()
        m.Va  = self._Va_var.get()
        m.Vf  = self._Vf_var.get()
        m.TL_rated = self._TL_var.get()
        m.T_amb    = self._Tamb_var.get()
        m.Rth_a    = self._Rtha_var.get()
        m.Cth_a    = self._Ctha_var.get()
        m.motor_type = self._motor_type_var.get()
        m.load_type  = self._load_type_var.get()
        m.ins_class  = self._ins_var.get()
        self._refresh_op_point()

    def _apply_config_and_reset(self):
        self._apply_config()
        self._stop_sim()
        self.model.reset()
        self._clear_history()
        self._refresh_op_point()

    def _refresh_op_point(self):
        ss = self.model.steady_state()
        T_max = INSULATION_CLASS.get(self.model.ins_class, 155)
        txt = (
            f"Typ silnika   : {self.model.motor_type}\n"
            f"Typ obciążenia: {self.model.load_type}\n"
            "─────────────────────────────────\n"
            f"Prąd znamionowy  Ia   = {ss['Ia_rated']:8.2f} A\n"
            f"Prędkość kąt.    ω    = {ss['omega_rated']:8.2f} rad/s\n"
            f"Prędkość obrot.  n    = {ss['RPM_rated']:8.0f} RPM\n"
            f"SEM wsteczna     Eb   = {ss['Eb']:8.2f} V\n"
            f"Strumień φ (norm)     = {ss['phi']:8.4f}\n"
            f"Moc wejściowa    P_in = {ss['P_in']:8.1f} W\n"
            f"Moc wyjściowa    P_out= {ss['P_out']:8.1f} W\n"
            f"Sprawność        η    = {ss['eta']:8.1f} %\n"
            "─────────────────────────────────\n"
            f"Va = {self.model.Va:.1f} V   Vf = {self.model.Vf:.1f} V\n"
            f"Ra = {self.model.Ra0:.3f} Ω  La = {self.model.La*1000:.1f} mH\n"
            f"Ke = {self.model.Ke:.3f}     Kt = {self.model.Kt:.3f}\n"
            f"J  = {self.model.J:.4f} kg·m²  B = {self.model.B:.4f}\n"
            f"Klasa izolacji: {self.model.ins_class}  (T_max = {T_max}°C)\n"
            f"I_trip = {self.model.I_trip:.1f} A\n"
            f"ω_max  = {self.model.omega_max:.1f} rad/s\n"
        )
        if hasattr(self, "_op_text"):
            self._op_text.delete("1.0", "end")
            self._op_text.insert("end", txt)

    # ================================================================
    # TAB 2 – Dynamic Simulation
    # ================================================================

    def _build_tab2_simulation(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="📈 Symulacja")

        # Control bar
        ctrl = ttk.Frame(tab)
        ctrl.pack(fill="x", padx=6, pady=4)
        ttk.Button(ctrl, text="▶ Start",  command=self._start_sim).pack(side="left", padx=3)
        ttk.Button(ctrl, text="⏸ Pause",  command=self._pause_sim).pack(side="left", padx=3)
        ttk.Button(ctrl, text="⏹ Stop",   command=self._stop_sim).pack(side="left", padx=3)
        ttk.Button(ctrl, text="🔄 Reset",  command=self._reset_sim).pack(side="left", padx=3)

        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=8)

        self._dist_var = tk.StringVar(value="Voltage Sag 30%")
        dist_options = [
            "Voltage Sag 30%", "Voltage Spike 20%",
            "Load Shock +100%", "Load Relief -50%",
            "Field Weakening 50%", "Field Loss",
            "Short Circuit", "Reversal (Va = -Va)",
            "Dynamic Braking", "Restore Normal",
        ]
        ttk.Combobox(ctrl, textvariable=self._dist_var,
                     values=dist_options, state="readonly", width=24).pack(side="left", padx=3)
        ttk.Button(ctrl, text="Zastosuj zakłócenie",
                   command=self._apply_disturbance).pack(side="left", padx=3)

        # 4 plots
        plot_frame = ttk.Frame(tab)
        plot_frame.pack(fill="both", expand=True)

        self._fig2 = Figure(figsize=(12, 6), dpi=90, tight_layout=True)
        self._ax_omega = self._fig2.add_subplot(2, 2, 1)
        self._ax_Ia    = self._fig2.add_subplot(2, 2, 2)
        self._ax_Te    = self._fig2.add_subplot(2, 2, 3)
        self._ax_Tarm  = self._fig2.add_subplot(2, 2, 4)
        for ax in [self._ax_omega, self._ax_Ia, self._ax_Te, self._ax_Tarm]:
            ax.grid(True, alpha=0.3)
        self._ax_omega.set_title("Prędkość ω(t)")
        self._ax_omega.set_ylabel("ω [rad/s]")
        self._ax_Ia.set_title("Prąd twornika Ia(t)")
        self._ax_Ia.set_ylabel("Ia [A]")
        self._ax_Te.set_title("Moment elektromagnetyczny Te(t)")
        self._ax_Te.set_ylabel("T [N·m]")
        self._ax_Tarm.set_title("Temperatura twornika T_arm(t)")
        self._ax_Tarm.set_ylabel("T [°C]")

        self._canvas2 = FigureCanvasTkAgg(self._fig2, plot_frame)
        self._canvas2.get_tk_widget().pack(fill="both", expand=True)

        # Status bar
        self._status_var = tk.StringVar(value="Gotowy.")
        ttk.Label(tab, textvariable=self._status_var,
                  relief="sunken", anchor="w").pack(fill="x", padx=6, pady=2)

    # ================================================================
    # TAB 3 – Torque-Speed Characteristics
    # ================================================================

    def _build_tab3_tn_curves(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="📊 Char. T-N")

        ctrl = ttk.LabelFrame(tab, text="Ustawienia")
        ctrl.pack(fill="x", padx=6, pady=6)

        row = ttk.Frame(ctrl)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Zakres napięcia (× znamionowe):", width=34).pack(side="left")
        self._tn_v_vars = []
        for v, default in [(0.5, True), (0.75, True), (1.0, True), (1.25, True)]:
            var = tk.BooleanVar(value=default)
            self._tn_v_vars.append((v, var))
            ttk.Checkbutton(row, text=f"{v}×", variable=var).pack(side="left", padx=6)

        row2 = ttk.Frame(ctrl)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Dodatkowa rezystancja twornika [Ω]:", width=34).pack(side="left")
        self._tn_Rext_var = tk.DoubleVar(value=0.0)
        self._make_slider(row2, "", self._tn_Rext_var, 0.0, 5.0, width=1)
        ttk.Button(ctrl, text="Rysuj krzywe T-N",
                   command=self._plot_tn_curves).pack(pady=6)

        fig_frame = ttk.Frame(tab)
        fig_frame.pack(fill="both", expand=True)
        self._fig3 = Figure(figsize=(12, 5), dpi=90, tight_layout=True)
        self._ax3a = self._fig3.add_subplot(1, 2, 1)
        self._ax3b = self._fig3.add_subplot(1, 2, 2)
        self._canvas3 = FigureCanvasTkAgg(self._fig3, fig_frame)
        self._canvas3.get_tk_widget().pack(fill="both", expand=True)

    def _plot_tn_curves(self):
        m = self.model
        self._apply_config()
        ax1, ax2 = self._ax3a, self._ax3b
        ax1.clear(); ax2.clear()
        ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
        ax1.set_title("Krzywa T-N (Moment – Prędkość)")
        ax1.set_xlabel("ω [rad/s]"); ax1.set_ylabel("Te [N·m]")
        ax2.set_title("Krzywa Ia-N (Prąd – Prędkość)")
        ax2.set_xlabel("ω [rad/s]"); ax2.set_ylabel("Ia [A]")

        omega_arr = np.linspace(0, m.omega_max * 1.1, 400)
        colors = ["blue", "green", "red", "orange"]
        R_ext = self._tn_Rext_var.get()

        for idx, (v_mult, var) in enumerate(self._tn_v_vars):
            if not var.get():
                continue
            Va = m.Va * v_mult
            label = f"Va = {Va:.0f} V"
            Te_list, Ia_list = [], []
            for w in omega_arr:
                phi = m.flux_phi(m.TL_rated / max(m.Kt, 1e-9))
                Eb = m.Ke * phi * w
                Ra_total = m.Ra0 + R_ext + (m.Rf_s if m.motor_type == "Series" else 0.0)
                if m.motor_type == "Series":
                    # iterative for series: phi depends on Ia
                    Ia = m.TL_rated / max(m.Kt * phi, 1e-9)  # initial guess
                    for _ in range(10):
                        phi = m.flux_phi(Ia)
                        Eb = m.Ke * phi * w
                        Ia = (Va - Eb) / max(Ra_total, 1e-9)
                else:
                    phi_val = m.Vf_eff() / m.Rf * m.Ke / (m.Ke * m.Vf / m.Rf)
                    Eb = m.Ke * phi_val * w
                    Ia = (Va - Eb) / max(Ra_total, 1e-9)
                    phi = phi_val
                Te = m.Kt * phi * Ia
                Te_list.append(max(Te, 0.0))
                Ia_list.append(Ia)
            ax1.plot(omega_arr, Te_list, color=colors[idx], label=label)
            ax2.plot(omega_arr, Ia_list, color=colors[idx], label=label)

        # Current operating point
        ss = m.steady_state()
        ax1.plot(ss["omega_rated"], m.TL_rated, "ko", markersize=8, label="Punkt pracy")
        ax2.plot(ss["omega_rated"], ss["Ia_rated"], "ko", markersize=8)
        ax1.axhline(m.TL_rated, color="gray", linestyle="--", alpha=0.5, label="TL znam.")
        ax1.legend(fontsize=8); ax2.legend(fontsize=8)
        self._canvas3.draw()

    # ================================================================
    # TAB 4 – Starting Methods
    # ================================================================

    def _build_tab4_starting(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="🚀 Rozruch")

        ctrl = ttk.LabelFrame(tab, text="Parametry rozruchu")
        ctrl.pack(fill="x", padx=6, pady=6)

        row = ttk.Frame(ctrl)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Metoda rozruchu:", width=24).pack(side="left")
        self._start_method_var = tk.StringVar(value="Direct On Line (DOL)")
        ttk.Combobox(row, textvariable=self._start_method_var,
                     values=["Direct On Line (DOL)", "Resistance Starting (3-stage)",
                              "PWM Soft Start", "No-Load Start"],
                     state="readonly", width=30).pack(side="left", padx=4)

        self._Rstart1_var = tk.DoubleVar(value=2.0)
        self._Rstart2_var = tk.DoubleVar(value=1.0)
        self._Rstart3_var = tk.DoubleVar(value=0.5)
        self._t_switch_var = tk.DoubleVar(value=0.5)
        self._pwm_ramp_var = tk.DoubleVar(value=1.0)
        self._make_slider(ctrl, "R_start1 [Ω]",        self._Rstart1_var, 0.0, 10.0, width=24)
        self._make_slider(ctrl, "R_start2 [Ω]",        self._Rstart2_var, 0.0,  8.0, width=24)
        self._make_slider(ctrl, "R_start3 [Ω]",        self._Rstart3_var, 0.0,  5.0, width=24)
        self._make_slider(ctrl, "Czas przełączenia [s]",self._t_switch_var,0.1,  2.0, width=24)
        self._make_slider(ctrl, "Czas rampy PWM [s]",  self._pwm_ramp_var, 0.1, 5.0, width=24)
        ttk.Button(ctrl, text="Symuluj rozruch",
                   command=self._simulate_starting).pack(pady=6)

        fig_frame = ttk.Frame(tab)
        fig_frame.pack(fill="both", expand=True)
        self._fig4 = Figure(figsize=(12, 5), dpi=90, tight_layout=True)
        self._ax4a = self._fig4.add_subplot(1, 2, 1)
        self._ax4b = self._fig4.add_subplot(1, 2, 2)
        self._canvas4 = FigureCanvasTkAgg(self._fig4, fig_frame)
        self._canvas4.get_tk_widget().pack(fill="both", expand=True)
        self._start_result_var = tk.StringVar(value="")
        ttk.Label(tab, textvariable=self._start_result_var,
                  relief="sunken", anchor="w").pack(fill="x", padx=6, pady=2)

    def _simulate_starting(self):
        """Run offline (non-live) starting method comparison."""
        self._apply_config()
        m = self.model
        T_sim = 4.0
        methods = {
            "DOL":       {"R": [0.0]},
            "Resistance":{
                "R": [self._Rstart1_var.get(),
                      self._Rstart2_var.get(),
                      self._Rstart3_var.get(), 0.0],
                "t_switch": self._t_switch_var.get()},
            "PWM":       {"ramp": self._pwm_ramp_var.get()},
            "No-Load":   {"R": [0.0]},
        }
        colors = {"DOL": "blue", "Resistance": "green", "PWM": "orange", "No-Load": "red"}
        ax1, ax2 = self._ax4a, self._ax4b
        ax1.clear(); ax2.clear()
        ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
        ax1.set_title("Prędkość podczas rozruchu")
        ax1.set_xlabel("t [s]"); ax1.set_ylabel("ω [rad/s]")
        ax2.set_title("Prąd twornika podczas rozruchu")
        ax2.set_xlabel("t [s]"); ax2.set_ylabel("Ia [A]")
        results = []

        for name, cfg in methods.items():
            Ia, omega, T_arm, T_fld = 0.0, 0.0, m.T_amb, m.T_amb
            t_arr, w_arr, I_arr = [], [], []
            t = 0.0
            peak_I = 0.0
            start_time = None
            TL_run = m.TL_rated if name != "No-Load" else 0.0
            stage = 0

            while t < T_sim:
                # Determine R_ext and Va_cmd
                if name == "DOL":
                    R_ext = 0.0
                    Va_cmd = m.Va
                elif name == "Resistance":
                    R_list = cfg["R"]
                    t_sw = cfg["t_switch"]
                    stage_now = min(int(t / t_sw), len(R_list) - 1)
                    R_ext = R_list[stage_now]
                    Va_cmd = m.Va
                elif name == "PWM":
                    ramp = cfg["ramp"]
                    Va_cmd = m.Va * min(t / ramp, 1.0)
                    R_ext = 0.0
                else:  # No-Load
                    R_ext = 0.0
                    Va_cmd = m.Va

                phi = m.flux_phi(Ia)
                Eb = m.Ke * phi * omega
                Ra_t = m.Ra_eff(T_arm)
                R_tot = Ra_t + R_ext + (m.Rf_s if m.motor_type == "Series" else 0.0)
                dIa = (Va_cmd - R_tot * Ia - Eb) / m.La
                Te = m.Kt * phi * Ia
                TL = m.load_torque(omega) * (TL_run / max(m.TL_rated, 1e-9)) if name != "No-Load" else 0.0
                dOmega = (Te - m.B * omega - TL) / m.J
                Ia     += dIa * DT
                omega  += dOmega * DT
                omega   = max(0.0, omega)
                t_arr.append(t)
                w_arr.append(omega)
                I_arr.append(Ia)
                peak_I = max(peak_I, abs(Ia))
                if start_time is None and omega > 0.9 * m.steady_state()["omega_rated"]:
                    start_time = t
                t += DT

            ax1.plot(t_arr[::10], w_arr[::10], color=colors[name], label=name)
            ax2.plot(t_arr[::10], I_arr[::10], color=colors[name], label=name)
            results.append(f"{name}: I_peak={peak_I:.1f}A, t_start={start_time:.2f}s" if start_time else f"{name}: I_peak={peak_I:.1f}A, t_start=>4s")

        ax1.legend(); ax2.legend()
        ax2.axhline(m.I_trip, color="red", linestyle="--", label="I_trip")
        self._canvas4.draw()
        self._start_result_var.set("  |  ".join(results))

    # ================================================================
    # TAB 5 – PID Controller
    # ================================================================

    def _build_tab5_pid(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="🎛 Regulator PID")

        ctrl = ttk.LabelFrame(tab, text="Nastawy regulatora PID")
        ctrl.pack(fill="x", padx=6, pady=6)

        self._pid_en_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Włącz regulator PID",
                        variable=self._pid_en_var,
                        command=self._apply_pid_config).pack(anchor="w", pady=4)

        self._pid_sp_var  = tk.DoubleVar(value=157.0)
        self._pid_kp_var  = tk.DoubleVar(value=5.0)
        self._pid_ki_var  = tk.DoubleVar(value=2.0)
        self._pid_kd_var  = tk.DoubleVar(value=0.05)
        self._pid_aw_var  = tk.DoubleVar(value=200.0)

        self._make_slider(ctrl, "Zadana prędkość [rad/s]", self._pid_sp_var, 0.0, 350.0, cmd=self._apply_pid_config)
        self._make_slider(ctrl, "Kp",                      self._pid_kp_var, 0.1,  20.0, cmd=self._apply_pid_config)
        self._make_slider(ctrl, "Ki",                      self._pid_ki_var, 0.0,  10.0, cmd=self._apply_pid_config)
        self._make_slider(ctrl, "Kd",                      self._pid_kd_var, 0.0,   1.0, cmd=self._apply_pid_config)
        self._make_slider(ctrl, "Anti-windup limit",       self._pid_aw_var, 10.0, 500.0, cmd=self._apply_pid_config)

        btn_row = ttk.Frame(ctrl)
        btn_row.pack(fill="x", pady=4)
        ttk.Button(btn_row, text="Auto-Tune (Ziegler-Nichols)",
                   command=self._zn_autotune).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Symuluj odpowiedź skokową",
                   command=self._pid_step_response).pack(side="left", padx=4)

        self._pid_result_var = tk.StringVar(value="")
        ttk.Label(ctrl, textvariable=self._pid_result_var,
                  relief="sunken", anchor="w").pack(fill="x", pady=2)

        fig_frame = ttk.Frame(tab)
        fig_frame.pack(fill="both", expand=True)
        self._fig5 = Figure(figsize=(12, 5), dpi=90, tight_layout=True)
        self._ax5a = self._fig5.add_subplot(1, 2, 1)
        self._ax5b = self._fig5.add_subplot(1, 2, 2)
        self._canvas5 = FigureCanvasTkAgg(self._fig5, fig_frame)
        self._canvas5.get_tk_widget().pack(fill="both", expand=True)

    def _apply_pid_config(self):
        m = self.model
        m.pid_enabled  = self._pid_en_var.get()
        m.pid_setpoint = self._pid_sp_var.get()
        m.Kp = self._pid_kp_var.get()
        m.Ki = self._pid_ki_var.get()
        m.Kd = self._pid_kd_var.get()
        m.pid_va_max = self._pid_aw_var.get()

    def _zn_autotune(self):
        """Estimate ultimate gain by bisection and compute ZN PID gains."""
        self._apply_config()
        m = self.model
        # Find Ku by simulating with only P control and finding oscillation
        # Simplified: use analytical approximation for DC motor
        # tau_m = J/(B)   tau_e = La/Ra
        tau_m = m.J / max(m.B, 1e-6)
        tau_e = m.La / max(m.Ra0, 1e-6)
        # Approx: Ku based on motor gain and time constants
        Km = m.Kt * m.Ke / max(m.Ra0 * m.J, 1e-9)
        Tu = 2 * math.pi * math.sqrt(tau_m * tau_e)
        Ku = 1.0 / (Km * tau_m) if Km * tau_m > 0 else 5.0
        Kp = 0.6 * Ku
        Ki = 2 * Kp / Tu
        Kd = Kp * Tu / 8
        self._pid_kp_var.set(round(Kp, 3))
        self._pid_ki_var.set(round(Ki, 3))
        self._pid_kd_var.set(round(Kd, 4))
        self._pid_result_var.set(f"ZN Auto-Tune: Ku={Ku:.2f}, Tu={Tu:.3f}s → Kp={Kp:.3f}, Ki={Ki:.3f}, Kd={Kd:.4f}")
        self._apply_pid_config()

    def _pid_step_response(self):
        """Offline PID step response simulation."""
        self._apply_config()
        self._apply_pid_config()
        m = self.model
        T_sim = 5.0
        sp = m.pid_setpoint
        Ia, omega = 0.0, 0.0
        T_arm = m.T_amb
        integral = 0.0
        prev_err  = sp
        t_arr, w_arr, Va_arr, err_arr = [], [], [], []
        t = 0.0
        while t < T_sim:
            err = sp - omega
            integral += err * DT
            integral = float(np.clip(integral, -m.pid_va_max / max(m.Ki, 1e-9),
                                     m.pid_va_max / max(m.Ki, 1e-9)))
            d_err = (err - prev_err) / DT
            prev_err = err
            Va_cmd = m.Kp * err + m.Ki * integral + m.Kd * d_err
            Va_cmd = float(np.clip(Va_cmd, 0.0, m.pid_va_max))
            phi = m.flux_phi(Ia)
            Eb = m.Ke * phi * omega
            Ra_t = m.Ra_eff(T_arm)
            dIa = (Va_cmd - Ra_t * Ia - Eb) / m.La
            Te = m.Kt * phi * Ia
            TL = m.load_torque(omega)
            dOmega = (Te - m.B * omega - TL) / m.J
            Ia     += dIa * DT
            omega  += dOmega * DT
            omega   = max(0.0, omega)
            t_arr.append(t)
            w_arr.append(omega)
            Va_arr.append(Va_cmd)
            err_arr.append(err)
            t += DT

        w_arr_np = np.array(w_arr)
        # Performance metrics
        overshoot = (w_arr_np.max() - sp) / sp * 100 if sp > 0 else 0
        rise_idx  = next((i for i, w in enumerate(w_arr) if w >= 0.9*sp), len(w_arr)-1)
        rise_time = t_arr[rise_idx]
        # settling: within 2%
        settle_idx = len(w_arr)-1
        for i in range(len(w_arr)-1, 0, -1):
            if abs(w_arr[i] - sp) > 0.02 * sp:
                settle_idx = i
                break
        settle_time = t_arr[settle_idx]
        ss_err = abs(w_arr[-1] - sp)

        ax1, ax2 = self._ax5a, self._ax5b
        ax1.clear(); ax2.clear()
        ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
        decimate = 5
        ax1.plot(t_arr[::decimate], w_arr[::decimate], "b-", label="ω(t)")
        ax1.axhline(sp, color="r", linestyle="--", label="Zadana")
        ax1.set_xlabel("t [s]"); ax1.set_ylabel("ω [rad/s]")
        ax1.set_title("Odpowiedź skokowa PID")
        ax1.legend()
        ax2.plot(t_arr[::decimate], Va_arr[::decimate], "g-", label="Va_cmd")
        ax2.plot(t_arr[::decimate], err_arr[::decimate], "r-", alpha=0.6, label="Błąd")
        ax2.set_xlabel("t [s]"); ax2.legend()
        ax2.set_title("Sygnał sterowania i błąd")
        self._canvas5.draw()
        self._pid_result_var.set(
            f"Przeregulowanie: {overshoot:.1f}%  "
            f"Czas narastania: {rise_time:.3f}s  "
            f"Czas ustalania: {settle_time:.3f}s  "
            f"Błąd ustalony: {ss_err:.2f} rad/s"
        )

    # ================================================================
    # TAB 6 – Disturbances & Scenarios
    # ================================================================

    def _build_tab6_disturbances(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="⚡ Zakłócenia")

        left = ttk.LabelFrame(tab, text="Ręczne zakłócenie")
        left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        right = ttk.LabelFrame(tab, text="Gotowe scenariusze")
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        # Manual disturbances
        self._va_dist_factor = tk.DoubleVar(value=0.7)
        self._va_dist_dur    = tk.DoubleVar(value=2.0)
        self._tl_dist_factor = tk.DoubleVar(value=2.0)
        self._tl_dist_dur    = tk.DoubleVar(value=2.0)
        self._vf_dist_factor = tk.DoubleVar(value=0.5)
        self._vf_dist_dur    = tk.DoubleVar(value=2.0)

        vaf = ttk.LabelFrame(left, text="Napięcie twornika Va")
        vaf.pack(fill="x", pady=4)
        self._make_slider(vaf, "Współczynnik Va (×)", self._va_dist_factor, 0.2, 2.0, width=26)
        self._make_slider(vaf, "Czas trwania [s]",    self._va_dist_dur,    0.1, 10.0, width=26)
        ttk.Button(vaf, text="Zastosuj Va",
                   command=lambda: self._manual_dist("Va")).pack()

        tfl = ttk.LabelFrame(left, text="Obciążenie TL")
        tfl.pack(fill="x", pady=4)
        self._make_slider(tfl, "Współczynnik TL (×)", self._tl_dist_factor, 0.0, 5.0, width=26)
        self._make_slider(tfl, "Czas trwania [s]",    self._tl_dist_dur,    0.1, 10.0, width=26)
        ttk.Button(tfl, text="Zastosuj TL",
                   command=lambda: self._manual_dist("TL")).pack()

        vff = ttk.LabelFrame(left, text="Wzbudzenie Vf")
        vff.pack(fill="x", pady=4)
        self._make_slider(vff, "Współczynnik Vf (×)", self._vf_dist_factor, 0.0, 1.5, width=26)
        self._make_slider(vff, "Czas trwania [s]",    self._vf_dist_dur,    0.1, 10.0, width=26)
        ttk.Button(vff, text="Zastosuj Vf",
                   command=lambda: self._manual_dist("Vf")).pack()

        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", pady=6)
        ttk.Button(btn_row, text="Zwarcie",         command=self._sc_fault).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Utrata wzbudzenia", command=self._field_loss).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Hamow. dynami.", command=self._dyn_braking).pack(side="left", padx=2)
        ttk.Button(btn_row, text="Rewersja",        command=self._reverse).pack(side="left", padx=2)

        # Scenarios
        scenarios = [
            "1. Normalny rozruch + obciążenie znamionowe",
            "2. Rozruch DOL (bezpośredni)",
            "3. Skok obciążenia +100%",
            "4. Zapad napięcia do 70% przez 2s",
            "5. Osłabienie pola do 50%",
            "6. Rewersja prędkości",
            "7. Hamowanie regeneracyjne",
            "8. Zablokowanie wirnika (praca zwarciowa)",
            "9. Utrata wzbudzenia (seria/bocznikowy)",
            "10. Przeciążenie termiczne",
        ]
        self._scenario_var = tk.StringVar(value=scenarios[0])
        ttk.Label(right, text="Wybierz scenariusz:").pack(anchor="w", padx=4, pady=4)
        lb_frame = ttk.Frame(right)
        lb_frame.pack(fill="both", expand=True, padx=4)
        lb_scroll = ttk.Scrollbar(lb_frame)
        lb_scroll.pack(side="right", fill="y")
        self._scenario_lb = tk.Listbox(lb_frame, yscrollcommand=lb_scroll.set, height=14)
        for s in scenarios:
            self._scenario_lb.insert("end", s)
        self._scenario_lb.select_set(0)
        self._scenario_lb.pack(fill="both", expand=True)
        lb_scroll.config(command=self._scenario_lb.yview)

        ttk.Button(right, text="▶ Uruchom scenariusz",
                   command=self._run_scenario).pack(pady=8)

        self._scenario_log = tk.Text(right, height=8, wrap="word",
                                     font=("Courier", 9))
        self._scenario_log.pack(fill="both", expand=True, padx=4, pady=4)

    def _manual_dist(self, kind: str):
        if not self.running:
            messagebox.showinfo("Info", "Uruchom symulację (Tab 2) przed zastosowaniem zakłócenia.")
            return
        m = self.model
        if kind == "Va":
            factor = self._va_dist_factor.get()
            dur    = self._va_dist_dur.get()
            m.Va_dist = factor
            self.root.after(int(dur * 1000), lambda: setattr(m, "Va_dist", 1.0))
        elif kind == "TL":
            factor = self._tl_dist_factor.get()
            dur    = self._tl_dist_dur.get()
            m.TL_dist = factor
            self.root.after(int(dur * 1000), lambda: setattr(m, "TL_dist", 1.0))
        elif kind == "Vf":
            factor = self._vf_dist_factor.get()
            dur    = self._vf_dist_dur.get()
            m.Vf_dist = factor
            self.root.after(int(dur * 1000), lambda: setattr(m, "Vf_dist", 1.0))

    def _sc_fault(self):
        m = self.model
        m.Va_dist = 3.0   # simulate short-circuit overvoltage / zero back-EMF effect
        m.R_start_val = 0.0

    def _field_loss(self):
        self.model.Vf_dist = 0.0

    def _dyn_braking(self):
        self.model.braking = True
        self.root.after(3000, lambda: setattr(self.model, "braking", False))

    def _reverse(self):
        self.model.Va_dist = -1.0
        self.root.after(4000, lambda: setattr(self.model, "Va_dist", 1.0))

    def _run_scenario(self):
        sel = self._scenario_lb.curselection()
        if not sel:
            return
        idx = sel[0]
        self._scenario_log.delete("1.0", "end")
        self._reset_sim()
        self._apply_config()
        m = self.model
        log = self._scenario_log

        def log_msg(msg):
            log.insert("end", msg + "\n")
            log.see("end")

        log_msg(f"Scenariusz {idx+1} uruchomiony …")

        if idx == 0:  # Normal start + rated load
            self._start_sim()
            log_msg("Silnik uruchomiony z obciążeniem znamionowym.")

        elif idx == 1:  # DOL
            m.R_start_val = 0.0
            self._start_sim()
            log_msg("Rozruch DOL – oczekuj wysokiego prądu szczytowego.")

        elif idx == 2:  # Load shock
            self._start_sim()
            def apply_shock():
                m.TL_dist = 2.0
                log_msg("Skok obciążenia +100% zastosowany.")
                self.root.after(3000, lambda: setattr(m, "TL_dist", 1.0))
            self.root.after(2000, apply_shock)

        elif idx == 3:  # Voltage sag
            self._start_sim()
            def sag():
                m.Va_dist = 0.7
                log_msg("Zapad napięcia do 70%.")
                self.root.after(2000, lambda: setattr(m, "Va_dist", 1.0))
            self.root.after(2000, sag)

        elif idx == 4:  # Field weakening
            self._start_sim()
            def fw():
                m.Vf_dist = 0.5
                log_msg("Osłabienie pola do 50% — prędkość wzrośnie.")
                self.root.after(3000, lambda: setattr(m, "Vf_dist", 1.0))
            self.root.after(2000, fw)

        elif idx == 5:  # Reversal
            self._start_sim()
            def rev():
                m.Va_dist = -1.0
                log_msg("Rewersja — zmiana znaku napięcia.")
                self.root.after(3000, lambda: setattr(m, "Va_dist", 1.0))
            self.root.after(2000, rev)

        elif idx == 6:  # Regenerative braking
            self._start_sim()
            def regen():
                m.TL_dist = -0.5  # negative → motor brakes
                log_msg("Hamowanie regeneracyjne — ujemne obciążenie.")
                self.root.after(3000, lambda: setattr(m, "TL_dist", 1.0))
            self.root.after(2000, regen)

        elif idx == 7:  # Stall
            self._start_sim()
            def stall():
                m.TL_dist = 20.0   # very high load → stall
                log_msg("Zablokowanie wirnika — obserwuj nagrzewanie!")
            self.root.after(1500, stall)

        elif idx == 8:  # Field loss
            self._start_sim()
            def fl():
                m.Vf_dist = 0.0
                log_msg("Utrata wzbudzenia — niebezpieczny wzrost prędkości!")
            self.root.after(2000, fl)

        elif idx == 9:  # Thermal overload cycle
            self._start_sim()
            def overload1():
                m.TL_dist = 2.5
                log_msg("Przeciążenie 250% — nagrzewanie…")
                self.root.after(3000, recover1)
            def recover1():
                m.TL_dist = 0.5
                log_msg("Odciążenie — chłodzenie…")
                self.root.after(2000, overload2)
            def overload2():
                m.TL_dist = 3.0
                log_msg("Kolejne przeciążenie 300%!")
            self.root.after(1000, overload1)

    # ================================================================
    # TAB 7 – Thermal Analysis
    # ================================================================

    def _build_tab7_thermal(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="🌡 Termika")

        ctrl = ttk.LabelFrame(tab, text="Parametry analizy termicznej")
        ctrl.pack(fill="x", padx=6, pady=6)

        self._th_duty_var = tk.DoubleVar(value=100.0)
        self._th_ol_var   = tk.DoubleVar(value=1.5)
        self._th_cool_var = tk.StringVar(value="Natural")
        self._make_slider(ctrl, "Cykl pracy [%]",         self._th_duty_var, 10.0, 100.0)
        self._make_slider(ctrl, "Współczynnik przeciążenia", self._th_ol_var, 1.0, 3.0)

        row = ttk.Frame(ctrl)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Chłodzenie:", width=24).pack(side="left")
        ttk.Combobox(row, textvariable=self._th_cool_var,
                     values=["Natural", "Forced (TEFC)", "Water Cooled"],
                     state="readonly", width=20).pack(side="left", padx=4)
        ttk.Button(ctrl, text="Oblicz krzywe termiczne",
                   command=self._plot_thermal).pack(pady=6)

        fig_frame = ttk.Frame(tab)
        fig_frame.pack(fill="both", expand=True)
        self._fig7 = Figure(figsize=(12, 5), dpi=90, tight_layout=True)
        self._ax7a = self._fig7.add_subplot(1, 3, 1)
        self._ax7b = self._fig7.add_subplot(1, 3, 2)
        self._ax7c = self._fig7.add_subplot(1, 3, 3)
        self._canvas7 = FigureCanvasTkAgg(self._fig7, fig_frame)
        self._canvas7.get_tk_widget().pack(fill="both", expand=True)
        self._thermal_info_var = tk.StringVar(value="")
        ttk.Label(tab, textvariable=self._thermal_info_var,
                  relief="sunken", anchor="w").pack(fill="x", padx=6, pady=2)

    def _plot_thermal(self):
        self._apply_config()
        m = self.model
        T_max = INSULATION_CLASS.get(m.ins_class, 155)
        cooling_factor = {"Natural": 1.0, "Forced (TEFC)": 0.5, "Water Cooled": 0.2}
        cf = cooling_factor.get(self._th_cool_var.get(), 1.0)
        Rth = m.Rth_a * cf
        tau_th = m.Cth_a * Rth

        ax1, ax2, ax3 = self._ax7a, self._ax7b, self._ax7c
        ax1.clear(); ax2.clear(); ax3.clear()
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.3)

        # Plot 1: T(t) for different overload factors
        t_arr = np.linspace(0, 5 * tau_th, 500)
        ss_rated = m.steady_state()
        P_rated = m.Ra0 * ss_rated["Ia_rated"] ** 2
        for ol in [1.0, 1.5, 2.0, 2.5]:
            P = P_rated * ol**2 * (self._th_duty_var.get() / 100.0)
            T_ss = m.T_amb + P * Rth
            T_t  = T_ss - (T_ss - m.T_amb) * np.exp(-t_arr / tau_th)
            ax1.plot(t_arr / tau_th, T_t, label=f"{ol}× obcią.")
        ax1.axhline(T_max, color="red", linestyle="--", label=f"T_max={T_max}°C")
        ax1.set_xlabel("t / τ_th"); ax1.set_ylabel("T [°C]")
        ax1.set_title("Nagrzewanie twornika")
        ax1.legend(fontsize=8)

        # Plot 2: Max overload time vs overload factor
        ol_arr = np.linspace(1.0, 3.0, 100)
        t_max_arr = []
        for ol in ol_arr:
            P = P_rated * ol**2
            T_ss = m.T_amb + P * Rth
            if T_ss <= T_max:
                t_max_arr.append(float("inf"))
            else:
                t_max_arr.append(-tau_th * math.log(max((T_ss - T_max) / (T_ss - m.T_amb), 1e-9)))
        t_max_arr_clamped = [min(t, 5*tau_th) for t in t_max_arr]
        ax2.plot(ol_arr, t_max_arr_clamped, "b-")
        ax2.set_xlabel("Współczynnik przeciążenia")
        ax2.set_ylabel("Maks. czas [s]")
        ax2.set_title("Dopuszczalny czas przeciążenia")

        # Plot 3: Derating vs ambient temperature
        T_amb_arr = np.linspace(0, 60, 100)
        derating = np.sqrt(np.clip((T_max - T_amb_arr) / (T_max - 25.0), 0, 1)) * 100
        ax3.plot(T_amb_arr, derating, "r-")
        ax3.set_xlabel("T_amb [°C]"); ax3.set_ylabel("% mocy znamionowej")
        ax3.set_title("Korekcja mocy vs. temperatura")
        ax3.axvline(25, color="gray", linestyle="--", alpha=0.7)

        self._canvas7.draw()
        self._thermal_info_var.set(
            f"τ_th = {tau_th:.1f} s  |  T_max (klasa {m.ins_class}) = {T_max}°C  |  "
            f"Chłodzenie: {self._th_cool_var.get()}  |  Rth_eff = {Rth:.2f} °C/W"
        )

    # ================================================================
    # TAB 8 – Protection & Faults
    # ================================================================

    def _build_tab8_protection(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="🛡 Zabezpieczenia")

        left = ttk.LabelFrame(tab, text="Nastawy zabezpieczeń")
        left.pack(side="left", fill="both", expand=False, padx=6, pady=6, ipadx=4)
        right = ttk.LabelFrame(tab, text="Logi i wykresy")
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        self._Itrip_var  = tk.DoubleVar(value=50.0)
        self._wmax_var   = tk.DoubleVar(value=350.0)
        self._Vmin_var   = tk.DoubleVar(value=80.0)
        self._i2t_var    = tk.DoubleVar(value=10000.0)

        self._make_slider(left, "I_trip [A]",       self._Itrip_var, 10.0, 200.0, cmd=self._apply_protection)
        self._make_slider(left, "ω_max [rad/s]",    self._wmax_var,  50.0, 500.0, cmd=self._apply_protection)
        self._make_slider(left, "V_min [V]",        self._Vmin_var,   0.0, 200.0, cmd=self._apply_protection)
        self._make_slider(left, "I²t limit [A²·s]", self._i2t_var, 1000.0, 100000.0, cmd=self._apply_protection)

        ttk.Button(left, text="Symuluj zwarcie", command=self._sim_fault).pack(pady=6)
        ttk.Button(left, text="Wyczyść log",     command=self._clear_trip_log).pack(pady=2)

        self._fault_result_var = tk.StringVar(value="")
        ttk.Label(left, textvariable=self._fault_result_var,
                  relief="sunken", anchor="w", wraplength=260).pack(fill="x", pady=4)

        fig_frame = ttk.Frame(right)
        fig_frame.pack(fill="both", expand=True)
        self._fig8 = Figure(figsize=(8, 4), dpi=90, tight_layout=True)
        self._ax8a = self._fig8.add_subplot(1, 2, 1)
        self._ax8b = self._fig8.add_subplot(1, 2, 2)
        self._canvas8 = FigureCanvasTkAgg(self._fig8, fig_frame)
        self._canvas8.get_tk_widget().pack(fill="both", expand=True)

        self._trip_log_text = tk.Text(right, height=10, wrap="word",
                                      font=("Courier", 9))
        self._trip_log_text.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Scrollbar(right, command=self._trip_log_text.yview).pack(side="right", fill="y")
        self._trip_log_text.configure(yscrollcommand=None)
        self._plot_protection_curves()

    def _apply_protection(self):
        m = self.model
        m.I_trip    = self._Itrip_var.get()
        m.omega_max = self._wmax_var.get()
        m.V_min     = self._Vmin_var.get()
        m.i2t_limit = self._i2t_var.get()

    def _plot_protection_curves(self):
        ax1, ax2 = self._ax8a, self._ax8b
        ax1.clear(); ax2.clear()
        ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
        I_arr = np.linspace(1, 200, 300)
        I_trip = self._Itrip_var.get() if hasattr(self, "_Itrip_var") else 50.0
        i2t_lim = self._i2t_var.get() if hasattr(self, "_i2t_var") else 10000.0
        t_relay = 0.14 / ((I_arr / I_trip)**0.02 - 1 + 1e-9)  # IEC inverse
        t_relay = np.clip(t_relay, 0.01, 100)
        ax1.plot(I_arr, t_relay, "b-", label="Przekaźnik I–t (IEC IDMT)")
        ax1.axvline(I_trip, color="red", linestyle="--", label=f"I_trip={I_trip:.0f}A")
        ax1.set_yscale("log"); ax1.set_xscale("log")
        ax1.set_xlabel("I [A]"); ax1.set_ylabel("t [s]")
        ax1.set_title("Charakterystyka I–t")
        ax1.legend(fontsize=8)
        # I²t curve
        t_i2t = i2t_lim / (I_arr**2)
        ax2.loglog(I_arr, t_i2t, "g-", label="I²t limit")
        ax2.axvline(I_trip, color="red", linestyle="--")
        ax2.set_xlabel("I [A]"); ax2.set_ylabel("t [s]")
        ax2.set_title("Granica I²t")
        ax2.legend(fontsize=8)
        if hasattr(self, "_canvas8"):
            self._canvas8.draw()

    def _sim_fault(self):
        """Simulate short-circuit from current operating state."""
        self._apply_config()
        self._apply_protection()
        m = self.model
        Ia, omega = m.Ia if self.running else 0.0, m.omega if self.running else 0.0
        T_arm = m.T_arm if self.running else m.T_amb
        # During short-circuit: Eb ~= 0 quickly, Va = m.Va
        Va_sc = m.Va
        Ra = m.Ra_eff(T_arm)
        I_peak = Va_sc / max(Ra, 1e-6)
        t_trip = m.i2t_limit / max(I_peak**2, 1e-9)
        E_diss = I_peak**2 * Ra * t_trip
        self._fault_result_var.set(
            f"I_szczyt = {I_peak:.1f} A  |  "
            f"t_wyłączenia = {t_trip*1000:.1f} ms  |  "
            f"E_rozproszone = {E_diss:.0f} J"
        )
        self._plot_protection_curves()

    def _clear_trip_log(self):
        self.model.trip_log.clear()
        self._trip_log_text.delete("1.0", "end")

    def _update_trip_log(self):
        m = self.model
        if m.trip_log:
            self._trip_log_text.delete("1.0", "end")
            self._trip_log_text.insert("end", "\n".join(m.trip_log))
            self._trip_log_text.see("end")

    # ================================================================
    # TAB 9 – Power Quality
    # ================================================================

    def _build_tab9_power_quality(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="⚡ Jakość energii")

        ctrl = ttk.LabelFrame(tab, text="Parametry")
        ctrl.pack(fill="x", padx=6, pady=6)

        self._pq_hours_var  = tk.DoubleVar(value=4000.0)
        self._pq_price_var  = tk.DoubleVar(value=0.65)
        self._make_slider(ctrl, "Godziny pracy/rok [h]", self._pq_hours_var, 100.0, 8760.0)
        self._make_slider(ctrl, "Cena energii [PLN/kWh]", self._pq_price_var, 0.1, 2.0)
        ttk.Button(ctrl, text="Oblicz widmo i sprawność",
                   command=self._plot_power_quality).pack(pady=6)

        fig_frame = ttk.Frame(tab)
        fig_frame.pack(fill="both", expand=True)
        self._fig9 = Figure(figsize=(12, 5), dpi=90, tight_layout=True)
        self._ax9a = self._fig9.add_subplot(1, 3, 1)
        self._ax9b = self._fig9.add_subplot(1, 3, 2)
        self._ax9c = self._fig9.add_subplot(1, 3, 3)
        self._canvas9 = FigureCanvasTkAgg(self._fig9, fig_frame)
        self._canvas9.get_tk_widget().pack(fill="both", expand=True)
        self._pq_info_var = tk.StringVar(value="")
        ttk.Label(tab, textvariable=self._pq_info_var,
                  relief="sunken", anchor="w").pack(fill="x", padx=6, pady=2)

    def _plot_power_quality(self):
        self._apply_config()
        m = self.model
        ss = m.steady_state()
        ax1, ax2, ax3 = self._ax9a, self._ax9b, self._ax9c
        ax1.clear(); ax2.clear(); ax3.clear()
        for ax in [ax1, ax2, ax3]:
            ax.grid(True, alpha=0.3)

        # Harmonic spectrum (approximate for PWM-driven DC motor)
        harmonics = [1, 3, 5, 7, 9, 11, 13]
        # relative amplitudes based on simple PWM ripple model
        amplitudes = [1.0, 0.04, 0.025, 0.015, 0.01, 0.008, 0.005]
        I_base = abs(ss["Ia_rated"]) if ss["Ia_rated"] else 1.0
        I_harm = [a * I_base for a in amplitudes]
        thd = math.sqrt(sum(h**2 for h in I_harm[1:])) / max(I_harm[0], 1e-9) * 100
        ax1.bar(harmonics, I_harm, color="steelblue", edgecolor="navy")
        ax1.set_xlabel("Rząd harmonicznej")
        ax1.set_ylabel("Amplituda [A]")
        ax1.set_title(f"Widmo harmoniczne (THD={thd:.1f}%)")

        # Efficiency map
        TL_arr = np.linspace(0.1, m.TL_rated * 1.5, 40)
        w_arr  = np.linspace(10, m.omega_max, 40)
        TL_g, w_g = np.meshgrid(TL_arr, w_arr)
        phi = ss["phi"]
        Ia_g = TL_g / max(m.Kt * phi, 1e-9)
        P_out_g = TL_g * w_g
        P_loss_g = m.Ra0 * Ia_g**2 + m.Vf**2/m.Rf
        P_in_g = P_out_g + P_loss_g
        eta_g = np.clip(P_out_g / np.where(P_in_g > 0, P_in_g, 1.0) * 100, 0, 100)
        cf = ax2.contourf(TL_arr, w_arr, eta_g, levels=20, cmap="RdYlGn")
        self._fig9.colorbar(cf, ax=ax2, label="η [%]")
        ax2.set_xlabel("TL [N·m]"); ax2.set_ylabel("ω [rad/s]")
        ax2.set_title("Mapa sprawności")
        # Operating point
        ax2.plot(m.TL_rated, ss["omega_rated"], "w*", markersize=12, label="Punkt pracy")
        ax2.legend()

        # Energy cost bar
        P_kW = ss["P_out"] / 1000.0
        annual_cost = P_kW / max(ss["eta"]/100, 1e-3) * self._pq_hours_var.get() * self._pq_price_var.get()
        ax3.bar(["Koszt roczny"], [annual_cost], color="tomato", edgecolor="darkred")
        ax3.set_ylabel("PLN/rok")
        ax3.set_title("Koszt energii (roczny)")
        for rect in ax3.patches:
            ax3.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01*annual_cost,
                     f"{annual_cost:.0f} PLN", ha="center", va="bottom", fontsize=10)

        self._canvas9.draw()

        P_loss = ss["P_in"] - ss["P_out"]
        self._pq_info_var.set(
            f"η = {ss['eta']:.1f}%  |  P_in = {ss['P_in']:.0f} W  |  "
            f"P_out = {ss['P_out']:.0f} W  |  P_strat = {P_loss:.0f} W  |  "
            f"THD = {thd:.1f}%  |  Koszt/rok ≈ {annual_cost:.0f} PLN"
        )

    # ================================================================
    # TAB 10 – Results & Summary
    # ================================================================

    def _build_tab10_results(self):
        tab = ttk.Frame(self.nb)
        self.nb.add(tab, text="📋 Wyniki")

        btn_row = ttk.Frame(tab)
        btn_row.pack(fill="x", padx=6, pady=6)
        ttk.Button(btn_row, text="🔄 Odśwież wyniki",
                   command=self._refresh_results).pack(side="left", padx=4)
        ttk.Button(btn_row, text="💾 Eksportuj CSV",
                   command=self._export_csv).pack(side="left", padx=4)

        self._result_text = tk.Text(tab, wrap="word", font=("Courier", 10))
        scr = ttk.Scrollbar(tab, command=self._result_text.yview)
        scr.pack(side="right", fill="y")
        self._result_text.configure(yscrollcommand=scr.set)
        self._result_text.pack(fill="both", expand=True, padx=6, pady=6)
        self._refresh_results()

    def _refresh_results(self):
        self._apply_config()
        m = self.model
        ss = m.steady_state()
        T_max = INSULATION_CLASS.get(m.ins_class, 155)
        tau_th = m.Cth_a * m.Rth_a
        trips = len(m.trip_log)

        # Simulation stats from history
        if self._hist_omega:
            w_arr = list(self._hist_omega)
            I_arr = list(self._hist_Ia)
            T_arr = list(self._hist_Tarm)
            w_min, w_max = min(w_arr), max(w_arr)
            I_min, I_max = min(I_arr), max(I_arr)
            T_a_max = max(T_arr)
            sim_time = self._hist_t[-1] - self._hist_t[0] if len(self._hist_t) > 1 else 0.0
        else:
            w_min = w_max = I_min = I_max = T_a_max = sim_time = 0.0

        notes = {
            "Separately Excited": "Moment startowy regulowany niezależnie. Doskonała regulacja prędkości.",
            "Shunt":              "Stała prędkość przy zmiennym obciążeniu. Niebezpieczna utrata wzbudzenia.",
            "Series":             "Duży moment startowy. Zakaz biegu luzem! Prędkość niekontrolowana.",
            "Compound Cumulative":"Łączy zalety bocznikowego i szeregowego. Stabilna praca.",
            "Compound Differential":"Niestabilny przy dużych obciążeniach. Stosowany rzadko.",
        }
        note = notes.get(m.motor_type, "")

        txt = (
            "══════════════════════════════════════════\n"
            "      WYNIKI SYMULACJI — SILNIK DC        \n"
            "══════════════════════════════════════════\n"
            f"\nTyp silnika         : {m.motor_type}\n"
            f"Typ obciążenia      : {m.load_type}\n"
            "\n--- PUNKT ZNAMIONOWY ---\n"
            f"Prąd twornika  Ia   = {ss['Ia_rated']:8.2f} A\n"
            f"Prędkość kąt.  ω    = {ss['omega_rated']:8.2f} rad/s\n"
            f"Prędkość obrot.n    = {ss['RPM_rated']:8.0f} RPM\n"
            f"Moc wejściowa  P_in = {ss['P_in']:8.1f} W\n"
            f"Moc wyjściowa  P_out= {ss['P_out']:8.1f} W\n"
            f"Sprawność      η    = {ss['eta']:8.1f} %\n"
            f"SEM wsteczna   Eb   = {ss['Eb']:8.2f} V\n"
            "\n--- PARAMETRY TERMICZNE ---\n"
            f"Klasa izolacji      : {m.ins_class}  (T_max = {T_max}°C)\n"
            f"Stała czasowa τ_th  = {tau_th:.1f} s\n"
            f"T_amb               = {m.T_amb:.1f} °C\n"
            "\n--- ZABEZPIECZENIA ---\n"
            f"I_trip              = {m.I_trip:.1f} A\n"
            f"ω_max               = {m.omega_max:.1f} rad/s\n"
            f"I²t limit           = {m.i2t_limit:.0f} A²·s\n"
            "\n--- STATYSTYKI SYMULACJI ---\n"
            f"Całk. czas symulacji= {sim_time:.2f} s\n"
            f"Liczba wyłączeń     = {trips}\n"
            f"ω: min={w_min:.1f}, max={w_max:.1f} rad/s\n"
            f"Ia: min={I_min:.1f}, max={I_max:.1f} A\n"
            f"T_arm_max           = {T_a_max:.1f} °C\n"
            "\n--- UWAGI INŻYNIERSKIE ---\n"
            f"{note}\n"
        )
        if m.trip_log:
            txt += "\n--- LOG WYŁĄCZEŃ ---\n" + "\n".join(m.trip_log[-10:]) + "\n"

        self._result_text.delete("1.0", "end")
        self._result_text.insert("end", txt)

    def _export_csv(self):
        path = "/tmp/dc_motor_log.csv"
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["t[s]", "omega[rad/s]", "Ia[A]", "Te[Nm]",
                                  "T_arm[C]", "Va_eff[V]"])
                for row in zip(self._hist_t, self._hist_omega, self._hist_Ia,
                               self._hist_Te, self._hist_Tarm, self._hist_Va):
                    writer.writerow([f"{v:.5f}" for v in row])
            messagebox.showinfo("Eksport", f"Zapisano {len(self._hist_t)} rekordów do:\n{path}")
        except Exception as exc:
            messagebox.showerror("Błąd eksportu", str(exc))

    # ================================================================
    # Disturbance helper (Tab 2 toolbar)
    # ================================================================

    def _apply_disturbance(self):
        if not self.running:
            messagebox.showinfo("Info", "Uruchom symulację przed zastosowaniem zakłócenia.")
            return
        m = self.model
        d = self._dist_var.get()
        if d == "Voltage Sag 30%":
            m.Va_dist = 0.7
            self.root.after(2000, lambda: setattr(m, "Va_dist", 1.0))
        elif d == "Voltage Spike 20%":
            m.Va_dist = 1.2
            self.root.after(500, lambda: setattr(m, "Va_dist", 1.0))
        elif d == "Load Shock +100%":
            m.TL_dist = 2.0
            self.root.after(2000, lambda: setattr(m, "TL_dist", 1.0))
        elif d == "Load Relief -50%":
            m.TL_dist = 0.5
            self.root.after(2000, lambda: setattr(m, "TL_dist", 1.0))
        elif d == "Field Weakening 50%":
            m.Vf_dist = 0.5
            self.root.after(3000, lambda: setattr(m, "Vf_dist", 1.0))
        elif d == "Field Loss":
            m.Vf_dist = 0.0
        elif d == "Short Circuit":
            m.Va_dist = 3.0
            m.R_start_val = 0.0
        elif d == "Reversal (Va = -Va)":
            m.Va_dist = -1.0
            self.root.after(4000, lambda: setattr(m, "Va_dist", 1.0))
        elif d == "Dynamic Braking":
            m.braking = True
            self.root.after(3000, lambda: setattr(m, "braking", False))
        elif d == "Restore Normal":
            m.Va_dist  = 1.0
            m.TL_dist  = 1.0
            m.Vf_dist  = 1.0
            m.braking  = False
            m.R_start_val = 0.0
            m.tripped  = False

    # ================================================================
    # Simulation loop
    # ================================================================

    def _start_sim(self):
        if self.model.tripped:
            messagebox.showwarning("Wyzwolony",
                "Silnik jest wyłączony (trip). Zresetuj symulację.")
            return
        if not self.running:
            self.running = True
            self._schedule_step()

    def _pause_sim(self):
        self.running = False
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _stop_sim(self):
        self.running = False
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _reset_sim(self):
        self._stop_sim()
        self.model.reset()
        self._clear_history()
        self._refresh_results()

    def _clear_history(self):
        for dq in [self._hist_t, self._hist_omega, self._hist_Ia,
                   self._hist_Te, self._hist_Tarm, self._hist_Va]:
            dq.clear()

    def _schedule_step(self):
        if self.running:
            self._sim_step()
            self._after_id = self.root.after(20, self._schedule_step)

    def _sim_step(self):
        m = self.model
        for _ in range(STEPS_PER_FRAME):
            if m.tripped:
                self.running = False
                break
            m.step()

        # Compute derived quantities
        phi = m.flux_phi(m.Ia)
        Te  = m.Kt * phi * m.Ia
        Va_cur = m.Va * m.Va_dist if not m.braking else 0.0

        # Store history
        self._hist_t.append(m.t)
        self._hist_omega.append(m.omega)
        self._hist_Ia.append(m.Ia)
        self._hist_Te.append(Te)
        self._hist_Tarm.append(m.T_arm)
        self._hist_Va.append(Va_cur)

        # Update plots
        self._update_live_plots()
        # Update status bar
        rpm = m.omega * 60 / (2 * math.pi)
        P_out = Te * m.omega
        P_in  = Va_cur * abs(m.Ia)
        eta   = P_out / max(P_in, 1.0) * 100 if P_in > 0 else 0
        status = (
            f"t={m.t:.2f}s  ω={m.omega:.1f} rad/s ({rpm:.0f} RPM)  "
            f"Ia={m.Ia:.2f} A  Te={Te:.2f} N·m  "
            f"Va={Va_cur:.1f} V  T_arm={m.T_arm:.1f}°C  "
            f"P_out={P_out:.0f}W  η={eta:.0f}%"
            + ("  [TRIP!]" if m.tripped else "")
        )
        self._status_var.set(status)

        # Update trip log display
        self._update_trip_log()

    def _update_live_plots(self):
        if not self._hist_t:
            return
        t   = list(self._hist_t)
        dec = max(1, len(t) // 500)   # decimate to keep plotting fast
        t   = t[::dec]
        w   = list(self._hist_omega)[::dec]
        Ia  = list(self._hist_Ia)[::dec]
        Te  = list(self._hist_Te)[::dec]
        Ta  = list(self._hist_Tarm)[::dec]

        m = self.model
        T_max = INSULATION_CLASS.get(m.ins_class, 155)

        for ax, data, color, label in [
            (self._ax_omega, w,  "royalblue",  "ω [rad/s]"),
            (self._ax_Ia,    Ia, "darkorange", "Ia [A]"),
            (self._ax_Te,    Te, "forestgreen","Te [N·m]"),
            (self._ax_Tarm,  Ta, "crimson",    "T_arm [°C]"),
        ]:
            ax.clear()
            ax.grid(True, alpha=0.3)
            ax.plot(t, data, color=color, linewidth=1.0)
            ax.set_xlabel("t [s]")
            ax.set_ylabel(label)

        # Reference lines
        self._ax_Ia.axhline(m.I_trip,  color="red", linestyle="--", linewidth=0.8, label="I_trip")
        self._ax_Ia.axhline(-m.I_trip, color="red", linestyle="--", linewidth=0.8)
        self._ax_Tarm.axhline(T_max, color="red", linestyle="--", linewidth=0.8, label=f"T_max={T_max}°C")
        self._ax_omega.axhline(m.omega_max, color="orange", linestyle="--", linewidth=0.8, label="ω_max")

        # Load torque reference
        TL = m.TL_rated * m.TL_dist
        self._ax_Te.axhline(TL, color="gray", linestyle=":", linewidth=0.8, label="TL")

        self._ax_omega.set_title("ω(t)")
        self._ax_Ia.set_title("Ia(t)")
        self._ax_Te.set_title("Te(t) / TL(t)")
        self._ax_Tarm.set_title("T_arm(t)")

        for ax in [self._ax_Ia, self._ax_Tarm, self._ax_omega, self._ax_Te]:
            ax.legend(fontsize=7, loc="upper right")

        self._canvas2.draw_idle()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = DCMotorApp(root)
    root.mainloop()
