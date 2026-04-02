"""
Double-Cage Induction Motor Comprehensive Engineering Suite (Tkinter).

This desktop app includes:
- Main menu and parameter sliders
- Differential-equation based electromechanical simulation
- Fault current module
- Protection coordination module
- Speed control module (PID + fuzzy gain scheduling)
- Thermal, economic, and harmonic/power-quality modules
- Start / Stop / Reset controls and auto-resizing plots
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

matplotlib.use("TkAgg")


class MotorModel:
    def __init__(self) -> None:
        self.e2 = 100.0
        self.r_in = 0.05
        self.x_in = 0.4
        self.r_out = 0.5
        self.x_out = 0.1
        self.freq = 50.0
        self.poles = 4
        self.J = 0.25
        self.B = 0.02
        self.t_load = 60.0

    @property
    def w_sync(self) -> float:
        return 4.0 * math.pi * self.freq / self.poles

    def cage_currents(self, slip: float) -> tuple[complex, complex]:
        s = float(np.clip(slip, 1e-4, 1.0))
        z_in = self.r_in / s + 1j * self.x_in
        z_out = self.r_out / s + 1j * self.x_out
        return self.e2 / z_in, self.e2 / z_out

    def torque_sync_watt_per_phase(self, slip: float) -> float:
        i_in, i_out = self.cage_currents(slip)
        s = float(np.clip(slip, 1e-4, 1.0))
        p_in = abs(i_in) ** 2 * (self.r_in / s)
        p_out = abs(i_out) ** 2 * (self.r_out / s)
        return p_in + p_out

    def torque_nm(self, slip: float) -> float:
        p_sync = self.torque_sync_watt_per_phase(slip)
        return 3.0 * p_sync / max(self.w_sync, 1e-6)


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Double-Cage IM Advanced Suite (Tkinter)")
        self.root.geometry("1320x860")

        self.model = MotorModel()
        self.running = False
        self.after_id: str | None = None

        self.time: list[float] = [0.0]
        self.speed: list[float] = [0.0]
        self.torque: list[float] = [0.0]

        self.slip_ref = tk.DoubleVar(value=0.05)
        self.step_load = tk.DoubleVar(value=80.0)
        self.pid_kp = tk.DoubleVar(value=4.0)
        self.pid_ki = tk.DoubleVar(value=6.0)
        self.pid_kd = tk.DoubleVar(value=0.02)
        self.e2_var = tk.DoubleVar(value=self.model.e2)
        self.rin_var = tk.DoubleVar(value=self.model.r_in)
        self.xin_var = tk.DoubleVar(value=self.model.x_in)
        self.rout_var = tk.DoubleVar(value=self.model.r_out)
        self.xout_var = tk.DoubleVar(value=self.model.x_out)

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True)

        self.tabs: dict[str, ttk.Frame] = {}
        for name in [
            "Main Menu",
            "Modeling + Simulation",
            "Fault Current",
            "Protection Coordination",
            "Speed Controller",
            "Thermal + Economic",
            "Harmonic + Energy Quality",
            "Results",
        ]:
            f = ttk.Frame(self.nb)
            self.nb.add(f, text=name)
            self.tabs[name] = f

        self._build_main_menu()
        self._build_model_tab()
        self._build_fault_tab()
        self._build_protection_tab()
        self._build_speed_tab()
        self._build_thermal_economic_tab()
        self._build_harmonic_tab()
        self._build_results_tab()
        self.refresh_all()

    def _slider(self, parent: ttk.Frame, label: str, var: tk.DoubleVar, lo: float, hi: float, step: float, cmd) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=32).pack(side="left")
        ttk.Scale(row, variable=var, from_=lo, to=hi, command=lambda _: cmd()).pack(side="left", fill="x", expand=True, padx=6)
        val_lbl = ttk.Label(row, width=8)
        val_lbl.pack(side="right")

        def update_label(*_args) -> None:
            val_lbl.configure(text=f"{var.get():.3g}")

        var.trace_add("write", update_label)
        update_label()

    def _build_main_menu(self) -> None:
        frame = self.tabs["Main Menu"]
        left = ttk.LabelFrame(frame, text="Input parameters + controls")
        left.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        right = ttk.LabelFrame(frame, text="Detailed explanation")
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        self._slider(left, "Rotor induced emf E2 (V/ph)", self.e2_var, 20, 300, 1, self.refresh_all)
        self._slider(left, "Inner cage R (Ω)", self.rin_var, 0.01, 1.0, 0.01, self.refresh_all)
        self._slider(left, "Inner cage X (Ω)", self.xin_var, 0.01, 1.0, 0.01, self.refresh_all)
        self._slider(left, "Outer cage R (Ω)", self.rout_var, 0.01, 2.0, 0.01, self.refresh_all)
        self._slider(left, "Outer cage X (Ω)", self.xout_var, 0.01, 1.0, 0.01, self.refresh_all)

        ctl = ttk.Frame(left)
        ctl.pack(fill="x", pady=10)
        ttk.Button(ctl, text="Start", command=self.start).pack(side="left", padx=3)
        ttk.Button(ctl, text="Stop", command=self.stop).pack(side="left", padx=3)
        ttk.Button(ctl, text="Reset", command=self.reset).pack(side="left", padx=3)

        self.main_text = tk.Text(right, wrap="word", height=25)
        self.main_text.pack(fill="both", expand=True, padx=6, pady=6)

    def _make_chart(self, parent: ttk.Frame, title: str) -> tuple[Figure, any, any]:
        fig = Figure(figsize=(6, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return fig, ax, canvas

    def _build_model_tab(self) -> None:
        frame = self.tabs["Modeling + Simulation"]
        self.fig_model, self.ax_model, self.canvas_model = self._make_chart(frame, "Differential Equation Simulation: ω(t)")

    def _build_fault_tab(self) -> None:
        frame = self.tabs["Fault Current"]
        self.fig_fault, self.ax_fault, self.canvas_fault = self._make_chart(frame, "Fault/Starting Current Envelope")

    def _build_protection_tab(self) -> None:
        frame = self.tabs["Protection Coordination"]
        self.fig_prot, self.ax_prot, self.canvas_prot = self._make_chart(frame, "Overcurrent Relay Coordination")

    def _build_speed_tab(self) -> None:
        frame = self.tabs["Speed Controller"]
        ctrl = ttk.LabelFrame(frame, text="Step load + controller tuning")
        ctrl.pack(fill="x", padx=8, pady=8)
        self._slider(ctrl, "Step load torque (N·m)", self.step_load, 10, 200, 1, self.refresh_all)
        self._slider(ctrl, "PID Kp", self.pid_kp, 0.1, 20, 0.1, self.refresh_all)
        self._slider(ctrl, "PID Ki", self.pid_ki, 0.1, 20, 0.1, self.refresh_all)
        self._slider(ctrl, "PID Kd", self.pid_kd, 0.0, 2.0, 0.01, self.refresh_all)
        self.fig_speed, self.ax_speed, self.canvas_speed = self._make_chart(frame, "PID vs Fuzzy Gain-Scheduled Response")

    def _build_thermal_economic_tab(self) -> None:
        frame = self.tabs["Thermal + Economic"]
        up = ttk.Frame(frame)
        up.pack(fill="both", expand=True)
        lo = ttk.Frame(frame)
        lo.pack(fill="both", expand=True)
        self.fig_th, self.ax_th, self.canvas_th = self._make_chart(up, "Thermal RC Model")
        self.fig_eco, self.ax_eco, self.canvas_eco = self._make_chart(lo, "Annual Energy Cost vs Efficiency")

    def _build_harmonic_tab(self) -> None:
        frame = self.tabs["Harmonic + Energy Quality"]
        self.fig_h, self.ax_h, self.canvas_h = self._make_chart(frame, "Harmonic Spectrum and THD")

    def _build_results_tab(self) -> None:
        frame = self.tabs["Results"]
        self.result_text = tk.Text(frame, wrap="word")
        self.result_text.pack(fill="both", expand=True, padx=8, pady=8)

    def start(self) -> None:
        if not self.running:
            self.running = True
            self._tick()

    def stop(self) -> None:
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def reset(self) -> None:
        self.stop()
        self.time, self.speed, self.torque = [0.0], [0.0], [0.0]
        self.refresh_all()

    def _tick(self) -> None:
        if not self.running:
            return
        dt = 0.02
        w = self.speed[-1]
        s = float(np.clip((self.model.w_sync - w) / self.model.w_sync, 1e-4, 1.0))
        te = self.model.torque_nm(s)
        tl = self.model.t_load if self.time[-1] > 1.0 else 0.6 * self.model.t_load
        dw = (te - tl - self.model.B * w) / max(self.model.J, 1e-6)
        w_next = float(np.clip(w + dt * dw, 0.0, 2.0 * self.model.w_sync))

        self.time.append(self.time[-1] + dt)
        self.speed.append(w_next)
        self.torque.append(te)
        self.update_model_plot()

        self.after_id = self.root.after(int(dt * 1000), self._tick)

    def refresh_all(self) -> None:
        self.model.e2 = float(np.clip(self.e2_var.get(), 1.0, 500.0))
        self.model.r_in = float(np.clip(self.rin_var.get(), 1e-3, 5.0))
        self.model.x_in = float(np.clip(self.xin_var.get(), 1e-3, 5.0))
        self.model.r_out = float(np.clip(self.rout_var.get(), 1e-3, 5.0))
        self.model.x_out = float(np.clip(self.xout_var.get(), 1e-3, 5.0))
        self.model.t_load = float(np.clip(self.step_load.get(), 1.0, 400.0))
        self.update_theory_text()
        self.update_model_plot()
        self.update_fault_plot()
        self.update_protection_plot()
        self.update_speed_plot()
        self.update_thermal_plot()
        self.update_economic_plot()
        self.update_harmonic_plot()
        self.update_result_text()

    def update_theory_text(self) -> None:
        p_st = self.model.torque_sync_watt_per_phase(1.0)
        p_5 = self.model.torque_sync_watt_per_phase(0.05)
        self.main_text.delete("1.0", "end")
        self.main_text.insert(
            "end",
            (
                "Double-cage motor torque in synchronous watts per phase:\n"
                "P_sync,ph = |I_in|^2 (R_in/s) + |I_out|^2 (R_out/s), with I = E2/(R/s + jX).\n\n"
                f"At standstill (s=1): {p_st:,.2f} synchronous W/phase\n"
                f"At 5% slip (s=0.05): {p_5:,.2f} synchronous W/phase\n\n"
                "Physical interpretation: outer cage dominates starting current due to higher R and low X, "
                "while inner cage contribution rises at low slip."
            ),
        )

    def update_model_plot(self) -> None:
        self.ax_model.clear()
        self.ax_model.plot(self.time, np.array(self.speed) * 60 / (2 * np.pi), lw=2, color="#2563eb")
        self.ax_model.set_xlabel("Time (s)")
        self.ax_model.set_ylabel("Speed (rpm)")
        self.ax_model.grid(True, alpha=0.3)
        self.canvas_model.draw_idle()

    def update_fault_plot(self) -> None:
        t = np.linspace(0, 0.2, 400)
        i_peak = np.clip(self.model.e2 / math.hypot(self.model.r_out, self.model.x_out), 0, 3000)
        i = i_peak * np.exp(-20 * t) * np.sin(2 * np.pi * 50 * t)
        self.ax_fault.clear()
        self.ax_fault.plot(t, i, color="#dc2626")
        self.ax_fault.set_xlabel("Time (s)")
        self.ax_fault.set_ylabel("Current (A)")
        self.ax_fault.grid(True, alpha=0.3)
        self.canvas_fault.draw_idle()

    def update_protection_plot(self) -> None:
        m = np.linspace(1.1, 20, 300)
        t_std = 0.14 / np.maximum(m**0.02 - 1.0, 1e-3)
        t_vi = 13.5 / np.maximum(m - 1.0, 1e-3)
        self.ax_prot.clear()
        self.ax_prot.semilogy(m, t_std, label="IEC Normal Inverse")
        self.ax_prot.semilogy(m, t_vi, label="IEC Very Inverse")
        self.ax_prot.set_xlabel("PSM = I/Is")
        self.ax_prot.set_ylabel("Trip time (s)")
        self.ax_prot.legend()
        self.ax_prot.grid(True, which="both", alpha=0.3)
        self.canvas_prot.draw_idle()

    def update_speed_plot(self) -> None:
        t = np.linspace(0, 6, 600)
        w_ref = self.model.w_sync * 0.95
        step_tl = np.where(t > 2, self.step_load.get(), 0.6 * self.step_load.get())

        pid = np.zeros_like(t)
        fuzzy = np.zeros_like(t)
        ei = 0.0
        e_prev = 0.0
        dt = t[1] - t[0]
        for k in range(1, len(t)):
            e = w_ref - pid[k - 1]
            ei = float(np.clip(ei + e * dt, -500, 500))
            ed = (e - e_prev) / dt
            u = self.pid_kp.get() * e + self.pid_ki.get() * ei + self.pid_kd.get() * ed
            pid[k] = np.clip(pid[k - 1] + dt * (u - step_tl[k]) / 50, 0, 2 * self.model.w_sync)
            e_prev = e

            ef = w_ref - fuzzy[k - 1]
            gain = 5.0 if abs(ef) > 15 else 2.0
            uf = gain * ef + 0.6 * (0 if k < 2 else (fuzzy[k - 1] - fuzzy[k - 2]) / dt)
            fuzzy[k] = np.clip(fuzzy[k - 1] + dt * (uf - step_tl[k]) / 55, 0, 2 * self.model.w_sync)

        self.ax_speed.clear()
        self.ax_speed.plot(t, pid * 60 / (2 * np.pi), label="PID", lw=2)
        self.ax_speed.plot(t, fuzzy * 60 / (2 * np.pi), label="Fuzzy", lw=2)
        self.ax_speed.axvline(2, ls="--", color="gray", label="Step load")
        self.ax_speed.set_xlabel("Time (s)")
        self.ax_speed.set_ylabel("Speed (rpm)")
        self.ax_speed.legend()
        self.ax_speed.grid(True, alpha=0.3)
        self.canvas_speed.draw_idle()

    def update_thermal_plot(self) -> None:
        t = np.linspace(0, 3600, 600)
        ta = 25.0
        p_loss = 4000.0
        rth, cth = 0.03, 90000.0
        temp = ta + p_loss * rth * (1 - np.exp(-t / (rth * cth)))
        self.ax_th.clear()
        self.ax_th.plot(t / 60, temp, color="#ea580c")
        self.ax_th.axhline(130, color="#f59e0b", ls="--", label="Class B")
        self.ax_th.axhline(155, color="#ef4444", ls="--", label="Class F")
        self.ax_th.set_xlabel("Time (min)")
        self.ax_th.set_ylabel("Winding temperature (°C)")
        self.ax_th.legend()
        self.ax_th.grid(True, alpha=0.3)
        self.canvas_th.draw_idle()

    def update_economic_plot(self) -> None:
        eff = np.linspace(0.8, 0.97, 60)
        p_out_kw = 75
        hrs = 6000
        price = 0.12
        pin = p_out_kw / eff
        annual = pin * hrs * price
        self.ax_eco.clear()
        self.ax_eco.plot(eff * 100, annual / 1000, color="#16a34a")
        self.ax_eco.set_xlabel("Efficiency (%)")
        self.ax_eco.set_ylabel("Annual energy cost (k$/year)")
        self.ax_eco.grid(True, alpha=0.3)
        self.canvas_eco.draw_idle()

    def update_harmonic_plot(self) -> None:
        orders = np.array([1, 5, 7, 11, 13])
        amps = np.array([100, 20, 14, 9, 7], dtype=float)
        thd = float(np.sqrt(np.sum(amps[1:] ** 2)) / max(amps[0], 1e-6) * 100)
        self.ax_h.clear()
        self.ax_h.bar(orders, amps, color="#7c3aed")
        self.ax_h.set_xlabel("Harmonic order")
        self.ax_h.set_ylabel("Current magnitude (%)")
        self.ax_h.set_title(f"THD = {thd:.2f}%")
        self.ax_h.grid(True, alpha=0.3)
        self.canvas_h.draw_idle()

    def update_result_text(self) -> None:
        p_st = self.model.torque_sync_watt_per_phase(1.0)
        p_5 = self.model.torque_sync_watt_per_phase(0.05)
        t_st = self.model.torque_nm(1.0)
        t_5 = self.model.torque_nm(0.05)
        self.result_text.delete("1.0", "end")
        self.result_text.insert(
            "end",
            (
                "Model validation snapshot\n"
                "------------------------\n"
                f"Standstill torque (sync W/ph): {p_st:,.2f}\n"
                f"5% slip torque (sync W/ph): {p_5:,.2f}\n"
                f"Equivalent shaft torque @standstill: {t_st:,.2f} N·m\n"
                f"Equivalent shaft torque @5% slip: {t_5:,.2f} N·m\n\n"
                "Checks:\n"
                "1) No divide-by-zero through minimum slip clipping (1e-4).\n"
                "2) Integrator anti-windup clamps in PID.\n"
                "3) Speed and current bounded to avoid overflow.\n"
            ),
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
