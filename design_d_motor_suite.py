#!/usr/bin/env python3
"""
Design D Induction Motor Suite
40 HP, 575 V, 60 Hz, 8-pole, 3-phase, Design D Induction Motor
with Steel Flywheel Load

Tabs:
  1  Overview / pre-computed answers
  2  Input Parameters (sliders + live results)
  3  Torque-Speed Curve
  4  Fault Current Modelling
  5  Protection Coordination (TCC)
  6  Speed Controller (PID / Fuzzy)
  7  Thermal Analysis
  8  Economic Analysis
  9  Harmonic & Power Quality
  10 Comprehensive Analysis (sensitivity + export)
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import time
import math

# ─── Default motor / flywheel parameters ─────────────────────────────────────
R1_DEF   = 0.30     # stator resistance  (Ω)
X1_DEF   = 0.60     # stator leakage reactance (Ω)
R2_DEF   = 0.65     # rotor resistance referred to stator (Ω)
X2_DEF   = 0.60     # rotor leakage reactance (Ω)
XM_DEF   = 35.0     # magnetising reactance (Ω)
VL_DEF   = 575.0    # line voltage (V)
FREQ_DEF = 60.0     # frequency (Hz)
POLES_DEF = 8       # poles
HP_DEF   = 40.0     # rated horsepower
FW_DIAM_DEF  = 31.5    # flywheel diameter (in)
FW_THICK_DEF = 7.875   # flywheel thickness (in)
STEEL_DENSITY_LB_IN3 = 0.2836   # lb/in³

# ─── Physical / model constants ───────────────────────────────────────────────
# Solid disk: I = ½·m·r²  →  WK² = ½·W·r²  (W in lb, r in ft → lb·ft²)
DISK_INERTIA_COEFF   = 0.5       # solid-disk moment of inertia coefficient
LB_FT2_TO_KG_M2      = 0.042140  # unit conversion: 1 lb·ft² = 0.042140 kg·m²

# Thermal model: assume steady-state temperature rise ≈ SS_TEMP_RISE_DEG °C
# at full load for a TEFC motor (Class F insulation, ambient 40 °C).
SS_TEMP_RISE_DEG     = 80.0   # °C — gives k_th = P_loss / SS_TEMP_RISE_DEG

# Speed controller: scale factor mapping PID/fuzzy output u → synchronous
# angular speed reference used to look up motor torque via the T(s) curve.
# u is bounded ±CTRL_U_MAX; dividing by (ws × CTRL_U_MAX) maps u to [0, 1].
CTRL_U_MAX           = 2000.0   # maximum control signal magnitude

# Rough empirical coefficient for estimating rotor/frame inertia from HP:
#   J_motor ≈ HP_INERTIA_COEFF × HP  [kg·m²]
# Derived from NEMA typical values for frame sizes in the 10–100 HP range.
HP_INERTIA_COEFF     = 0.05

DARK_BG  = '#1e1e2e'
DARK_AX  = '#181825'
CLR_TEXT = '#cdd6f4'
CLR_TITLE= '#89dceb'
CLR_RED  = '#f38ba8'
CLR_GRN  = '#a6e3a1'
CLR_ORG  = '#fab387'
CLR_PUR  = '#cba6f7'
CLR_GRID = '#313244'
CLR_SPINE= '#45475a'


# ─── Helper: dark-style axis ──────────────────────────────────────────────────
def _style_ax(ax, title='', xlabel='', ylabel='', legend=True):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=CLR_TEXT)
    ax.spines[:].set_color(CLR_SPINE)
    ax.grid(True, color=CLR_GRID, linestyle='--', alpha=0.7)
    if title:
        ax.set_title(title, color=CLR_TITLE, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=CLR_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, color=CLR_TEXT)
    if legend:
        try:
            ax.legend(facecolor='#313244', edgecolor=CLR_SPINE, labelcolor=CLR_TEXT,
                      fontsize=8)
        except Exception:
            pass


# ─── Main application class ───────────────────────────────────────────────────
class DesignDMotorSuite:
    def __init__(self, root):
        self.root = root
        self.root.title(
            "Design D Induction Motor Suite  —  40 HP / 575 V / 60 Hz / 8-Pole")
        self.root.geometry("1300x820")
        self.root.minsize(900, 600)

        # ── Tk variables ──────────────────────────────────────────────────────
        self.var_VL       = tk.DoubleVar(value=VL_DEF)
        self.var_HP       = tk.DoubleVar(value=HP_DEF)
        self.var_freq     = tk.DoubleVar(value=FREQ_DEF)
        self.var_poles    = tk.IntVar(value=POLES_DEF)
        self.var_R1       = tk.DoubleVar(value=R1_DEF)
        self.var_X1       = tk.DoubleVar(value=X1_DEF)
        self.var_R2       = tk.DoubleVar(value=R2_DEF)
        self.var_X2       = tk.DoubleVar(value=X2_DEF)
        self.var_Xm       = tk.DoubleVar(value=XM_DEF)
        self.var_fw_diam  = tk.DoubleVar(value=FW_DIAM_DEF)
        self.var_fw_thick = tk.DoubleVar(value=FW_THICK_DEF)

        # ── Threading ─────────────────────────────────────────────────────────
        self.sim_running  = threading.Event()
        self._canvases    = []
        self._figures     = []

        # ── Notebook ──────────────────────────────────────────────────────────
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill='both', expand=True)

        self._build_tab1_overview()
        self._build_tab2_inputs()
        self._build_tab3_torquespeed()
        self._build_tab4_fault()
        self._build_tab5_protection()
        self._build_tab6_speed_ctrl()
        self._build_tab7_thermal()
        self._build_tab8_economic()
        self._build_tab9_harmonics()
        self._build_tab10_comprehensive()

        self.root.bind('<Configure>', self._on_resize)

    # ══════════════════════════════════════════════════════════════════════════
    #  MOTOR PHYSICS
    # ══════════════════════════════════════════════════════════════════════════
    def _thevenin(self, VL=None, R1=None, X1=None, Xm=None, freq=None, poles=None):
        VL    = VL    if VL    is not None else self.var_VL.get()
        R1    = R1    if R1    is not None else self.var_R1.get()
        X1    = X1    if X1    is not None else self.var_X1.get()
        Xm    = Xm    if Xm    is not None else self.var_Xm.get()
        freq  = freq  if freq  is not None else self.var_freq.get()
        poles = poles if poles is not None else self.var_poles.get()

        Vph = VL / math.sqrt(3.0)
        Ns  = 120.0 * freq / poles
        ws  = 2.0 * math.pi * Ns / 60.0

        Z1  = complex(R1, X1)
        Zm  = complex(0.0, Xm)
        Zth = (Z1 * Zm) / (Z1 + Zm)
        Rth = Zth.real
        Xth = Zth.imag

        Vth = Vph * Xm / abs(Z1 + Zm)
        return Vth, Rth, Xth, Ns, ws

    def _torque_at_slip(self, s,
                        VL=None, R1=None, X1=None,
                        R2=None, X2=None, Xm=None,
                        freq=None, poles=None):
        if s <= 0:
            return 0.0
        VL    = VL    if VL    is not None else self.var_VL.get()
        R1    = R1    if R1    is not None else self.var_R1.get()
        X1    = X1    if X1    is not None else self.var_X1.get()
        R2    = R2    if R2    is not None else self.var_R2.get()
        X2    = X2    if X2    is not None else self.var_X2.get()
        Xm    = Xm    if Xm    is not None else self.var_Xm.get()
        freq  = freq  if freq  is not None else self.var_freq.get()
        poles = poles if poles is not None else self.var_poles.get()

        Vth, Rth, Xth, Ns, ws = self._thevenin(VL, R1, X1, Xm, freq, poles)
        Xt  = Xth + X2
        R2s = R2 / s
        T   = (3.0 / ws) * Vth**2 * R2s / ((Rth + R2s)**2 + Xt**2)
        return T

    def _torque_speed_curve(self, num=600):
        Vth, Rth, Xth, Ns, ws = self._thevenin()
        X2 = self.var_X2.get()
        R2 = self.var_R2.get()
        Xt = Xth + X2
        slips  = np.linspace(1e-4, 1.0, num)
        speeds = Ns * (1.0 - slips)
        R2s    = R2 / slips
        T      = (3.0 / ws) * Vth**2 * R2s / ((Rth + R2s)**2 + Xt**2)
        return speeds, T, Ns

    def _rated_quantities(self):
        HP    = self.var_HP.get()
        freq  = self.var_freq.get()
        poles = self.var_poles.get()
        Ns    = 120.0 * freq / poles
        ws    = 2.0 * math.pi * Ns / 60.0
        # Bisection to find s where P_mech = HP*746
        target = HP * 746.0 / ws  # T*(1-s) = target  (approx)
        slo, shi = 1e-4, 0.999
        for _ in range(80):
            smid  = (slo + shi) / 2.0
            T_mid = self._torque_at_slip(smid)
            Pm    = T_mid * (1.0 - smid)
            if Pm > target:
                slo = smid
            else:
                shi = smid
        s_fl = (slo + shi) / 2.0
        Nr   = Ns * (1.0 - s_fl)
        wr   = ws * (1.0 - s_fl)
        T_Nm = HP * 746.0 / wr if wr > 0 else 0.0
        T_ft = T_Nm / 1.35582
        return Ns, Nr, s_fl, T_Nm, T_ft

    def _flywheel_calc(self):
        d_in = self.var_fw_diam.get()
        t_in = self.var_fw_thick.get()
        r_in = d_in / 2.0
        vol_in3   = math.pi * r_in**2 * t_in
        weight_lb = vol_in3 * STEEL_DENSITY_LB_IN3
        r_ft  = r_in / 12.0
        WK2   = DISK_INERTIA_COEFF * weight_lb * r_ft**2   # lb·ft²  (solid disk: ½Wr²)
        J_SI  = WK2 * LB_FT2_TO_KG_M2                      # kg·m²
        return r_in, weight_lb, WK2, J_SI

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 1 — Overview
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab1_overview(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='1: Overview')
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        st = scrolledtext.ScrolledText(
            frame, font=('Courier', 11), wrap='word',
            bg=DARK_BG, fg=CLR_TEXT, insertbackground='white')
        st.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        st.tag_config('title',   font=('Courier', 14, 'bold'), foreground=CLR_TITLE)
        st.tag_config('header',  font=('Courier', 12, 'bold'), foreground=CLR_RED)
        st.tag_config('formula', font=('Courier', 11),         foreground=CLR_GRN)
        st.tag_config('result',  font=('Courier', 11, 'bold'), foreground=CLR_ORG)
        st.tag_config('normal',  font=('Courier', 11),         foreground=CLR_TEXT)
        st.tag_config('note',    font=('Courier', 10, 'italic'),foreground='#bac2de')

        def ins(text, tag='normal'):
            st.insert('end', text, tag)

        ins('═' * 72 + '\n', 'title')
        ins('  DESIGN D INDUCTION MOTOR ANALYSIS SUITE\n', 'title')
        ins('  40 HP | 575 V | 60 Hz | 8-Pole | Steel Flywheel Load\n', 'title')
        ins('═' * 72 + '\n\n', 'title')

        ins('PROBLEM SPECIFICATION\n', 'header')
        ins('─' * 60 + '\n', 'header')
        ins('  Motor  : 40 HP, 575 V, 60 Hz, 8-pole, 3-phase, Design D\n')
        ins('  Load   : Solid steel flywheel, Ø=31.5 in, t=7.875 in\n')
        ins('  Class  : Design D — high rotor resistance → peak torque near stall\n')
        ins('  LRT    : ≈ 275% of rated torque\n\n')

        ins('MOTOR EQUIVALENT CIRCUIT PARAMETERS\n', 'header')
        ins('─' * 60 + '\n', 'header')
        ins('  R1 = 0.30 Ω      X1 = 0.60 Ω\n', 'formula')
        ins('  R2 = 0.65 Ω      X2 = 0.60 Ω     (R2/X2 ≈ 1.08  → Design D)\n',
            'formula')
        ins('  Xm = 35.0 Ω      (magnetising reactance)\n', 'formula')
        ins('  VL = 575 V   f = 60 Hz   P = 8 poles\n\n', 'formula')

        ins('PRE-COMPUTED ANSWERS\n', 'header')
        ins('─' * 60 + '\n', 'header')

        ins('\na) FLYWHEEL INERTIA (WK²)\n', 'header')
        ins('   ρ_steel = 490 lb/ft³ = 0.2836 lb/in³\n')
        ins('   r       = 31.5/2 = 15.75 in = 1.3125 ft\n')
        ins('   L       = 7.875 in = 0.65625 ft\n')
        ins('   Volume  = π × (1.3125)² × 0.65625 = 3.5534 ft³\n', 'formula')
        ins('   Weight W= 3.5534 × 490           = 1741.2 lb\n', 'result')
        ins('   WK²     = ½ × W × r²\n', 'formula')
        ins('           = ½ × 1741.2 × (1.3125)²\n', 'formula')
        ins('           = ½ × 1741.2 × 1.7227\n', 'formula')
        ins('   WK²     = 1498.5 lb·ft²\n', 'result')
        ins('   J_SI    = 1498.5 × 0.04214 = 63.15 kg·m²\n', 'result')

        ins('\nb) RATED SPEED & TORQUE\n', 'header')
        ins('   Ns      = 120×60/8          = 900 rpm  (sync speed)\n', 'formula')
        ins('   s_fl    ≈ 7.2%              (from equivalent circuit)\n')
        ins('   Nr      = 900×(1−0.072)    ≈ 835 rpm\n', 'result')
        ins('   T_rated = HP×5252/Nr\n', 'formula')
        ins('           = 40×5252/835      = 251.6 ft·lbf\n', 'result')
        ins('           = 251.6×1.35582    = 341.2 N·m\n', 'result')

        ins('\nc) LOCKED-ROTOR TORQUE (s = 1)\n', 'header')
        ins('   T_LR = 275% × T_rated\n', 'formula')
        ins('        = 2.75 × 251.6       ≈ 692 ft·lbf\n', 'result')
        ins('        = 692 × 1.35582      ≈ 938 N·m\n', 'result')

        ins('\nd) TORQUES AT SELECTED SPEEDS — from Thevenin model\n', 'header')
        ins('   Vth ≈ 325 V,  Rth ≈ 0.289 Ω,  Xt = Xth+X2 ≈ 1.2 Ω\n', 'formula')
        ins('   ws  = 2π×900/60 = 94.25 rad/s\n', 'formula')
        ins('   T(s) = (3/ws)·Vth²·(R2/s) / [(Rth+R2/s)² + Xt²]\n\n', 'formula')

        # Compute table
        Vth, Rth, Xth, Ns_def, ws_def = self._thevenin()
        X2_def = self.var_X2.get()
        R2_def = self.var_R2.get()
        Xt_def = Xth + X2_def

        ins('   Speed(rpm)  Slip(%)   T (N·m)   T (ft·lbf)\n', 'header')
        ins('   ' + '─' * 46 + '\n', 'header')
        for spd in [0, 180, 360, 540, 720, 810]:
            s_v = (Ns_def - spd) / Ns_def if spd < Ns_def else 1.0
            if s_v <= 0:
                T_Nm = 0.0
            else:
                R2s  = R2_def / s_v
                T_Nm = (3.0/ws_def) * Vth**2 * R2s / ((Rth + R2s)**2 + Xt_def**2)
            T_ft = T_Nm / 1.35582
            ins(f'   {spd:6d}      {s_v*100:6.2f}     {T_Nm:8.1f}   {T_ft:8.1f}\n',
                'result')

        ins('\n\nDESIGN D THEORY\n', 'header')
        ins('─' * 60 + '\n', 'header')
        ins(
            '\n  Design D motors have very high rotor resistance (R2/X2 ≥ 1).\n'
            '  This shifts the breakdown (peak) torque to near s = 1, so:\n\n'
            '    • LRT ≈ 275% rated  (vs ~150% for Design B)\n'
            '    • No distinct pull-out peak in the running region\n'
            '    • Higher full-load slip: 5–8%  (vs 1–3% for Design B)\n'
            '    • Monotonically decreasing T-N curve from standstill\n'
            '    • Ideal for punch-presses, hoists, flywheel-driven loads\n\n'
            '  Thevenin equivalent per phase:\n'
            '  ┌──Rth──jXth──┬──jX2──R2/s──┐\n'
            '  Vth           jXm            │\n'
            '  └─────────────┴─────────────┘\n'
            '\n  The slip at max torque: s_pk = R2 / √(Rth² + (Xth+X2)²)\n'
            '  For Design D this is close to 1.0.\n',
            'note')

        ins('\n' + '═' * 72 + '\n', 'title')
        st.configure(state='disabled')

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 2 — Input Parameters
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab2_inputs(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='2: Inputs')
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(0, weight=1)

        # ── left: sliders ─────────────────────────────────────────────────────
        left = ttk.LabelFrame(frame, text='Motor & Flywheel Parameters')
        left.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        left.columnconfigure(1, weight=1)

        slider_defs = [
            ('Line Voltage VL (V)',      self.var_VL,       400,   750, 1.0),
            ('Power HP',                 self.var_HP,        10,   100, 1.0),
            ('Frequency (Hz)',           self.var_freq,      50,    70, 0.5),
            ('Poles',                    self.var_poles,      4,    12, 2),
            ('R1 (Ω)',                   self.var_R1,      0.05,  1.50, 0.01),
            ('X1 (Ω)',                   self.var_X1,      0.10,  2.00, 0.05),
            ('R2 (Ω)',                   self.var_R2,      0.10,  2.00, 0.05),
            ('X2 (Ω)',                   self.var_X2,      0.10,  2.00, 0.05),
            ('Xm (Ω)',                   self.var_Xm,      10.0,  80.0, 0.5),
            ('Flywheel Diameter (in)',   self.var_fw_diam,  10.0,  60.0, 0.5),
            ('Flywheel Thickness (in)',  self.var_fw_thick,  2.0,  20.0, 0.25),
        ]

        self._slider_label_map = {}
        for row_idx, (label, var, lo, hi, _res) in enumerate(slider_defs):
            ttk.Label(left, text=label).grid(
                row=row_idx, column=0, sticky='w', padx=4, pady=2)
            sl = ttk.Scale(left, from_=lo, to=hi, variable=var, orient='horizontal')
            sl.grid(row=row_idx, column=1, sticky='ew', padx=4, pady=2)
            lbl = ttk.Label(left, text=f'{var.get():.3g}', width=8,
                            anchor='w')
            lbl.grid(row=row_idx, column=2, sticky='w', padx=2)
            self._slider_label_map[id(var)] = (var, lbl)
            var.trace_add('write', self._slider_changed)

        btn_row = len(slider_defs)
        ttk.Button(left, text='Recalculate',
                   command=self._update_input_results).grid(
            row=btn_row, column=0, columnspan=3, pady=8)

        # ── right: live results ───────────────────────────────────────────────
        right = ttk.LabelFrame(frame, text='Live Results')
        right.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._inp_results = scrolledtext.ScrolledText(
            right, font=('Courier', 11), bg=DARK_BG, fg=CLR_TEXT,
            insertbackground='white', state='disabled')
        self._inp_results.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        self._update_input_results()

    def _slider_changed(self, *_args):
        for _vid, (var, lbl) in self._slider_label_map.items():
            try:
                lbl.config(text=f'{var.get():.3g}')
            except Exception:
                pass
        self._update_input_results()

    def _update_input_results(self, *_args):
        try:
            Ns, Nr, s_fl, T_Nm, T_ft = self._rated_quantities()
            r_in, weight_lb, WK2, J_SI = self._flywheel_calc()
            Vth, Rth, Xth, _Ns, ws = self._thevenin()
            X2 = self.var_X2.get()
            R2 = self.var_R2.get()
            Xt = Xth + X2
            T_LR = self._torque_at_slip(1.0)
            s_pk = R2 / math.sqrt(max(Rth**2 + Xt**2, 1e-12))
            T_pk = self._torque_at_slip(s_pk)

            lines = [
                '═' * 54,
                '  MOTOR PERFORMANCE — LIVE RESULTS',
                '═' * 54,
                f'  Synchronous speed  Ns  = {Ns:.1f} rpm',
                f'  Full-load slip   s_fl  = {s_fl*100:.2f} %',
                f'  Rated speed       Nr   = {Nr:.1f} rpm',
                f'  T_rated               = {T_Nm:.1f} N·m  ({T_ft:.1f} ft·lbf)',
                '',
                f'  Locked-rotor torque   = {T_LR:.1f} N·m  ({T_LR/1.35582:.1f} ft·lbf)',
                f'  T_LR / T_rated        = {T_LR/T_Nm*100:.1f} %',
                '',
                f'  Peak torque (s={s_pk:.3f}) = {T_pk:.1f} N·m',
                '',
                '  Thevenin Equivalent:',
                f'    Vth = {Vth:.3f} V',
                f'    Rth = {Rth:.5f} Ω',
                f'    Xth = {Xth:.5f} Ω',
                f'    Xt  = Xth+X2 = {Xt:.4f} Ω',
                '',
                '─' * 54,
                '  FLYWHEEL',
                '─' * 54,
                f'  Diameter = {self.var_fw_diam.get():.2f} in',
                f'  Radius r = {r_in:.4f} in = {r_in/12:.4f} ft',
                f'  Weight W = {weight_lb:.1f} lb',
                f'  WK²      = {WK2:.1f} lb·ft²',
                f'  J (SI)   = {J_SI:.3f} kg·m²',
                '',
                '─' * 54,
                '  TORQUES AT SELECTED SPEEDS',
                '─' * 54,
                f"  {'Speed':>8}  {'Slip%':>7}  {'T (N·m)':>9}  {'T (ft·lbf)':>12}",
                '  ' + '─' * 42,
            ]
            for spd in [0, 180, 360, 540, 720, 810, int(Nr)]:
                sv = (Ns - spd) / Ns if spd < Ns else 1.0
                Tv = 0.0 if sv <= 0 else self._torque_at_slip(sv)
                Tf = Tv / 1.35582
                lines.append(
                    f'  {spd:8d}  {sv*100:7.2f}  {Tv:9.1f}  {Tf:12.1f}')
            lines.append('═' * 54)

            txt = self._inp_results
            txt.configure(state='normal')
            txt.delete('1.0', 'end')
            txt.insert('end', '\n'.join(lines))
            txt.configure(state='disabled')
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 3 — Torque-Speed Curve
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab3_torquespeed(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='3: T-N Curve')
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=3)
        frame.rowconfigure(1, weight=0)
        frame.rowconfigure(2, weight=1)

        # Figure
        self.fig3, self.ax3 = plt.subplots(figsize=(9, 5))
        self.fig3.patch.set_facecolor(DARK_BG)
        canvas3 = FigureCanvasTkAgg(self.fig3, master=frame)
        canvas3.get_tk_widget().grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
        self._canvases.append(canvas3)
        self._figures.append(self.fig3)
        self.canvas3 = canvas3

        # Button row
        btn_row = ttk.Frame(frame)
        btn_row.grid(row=1, column=0, sticky='ew', padx=6)
        ttk.Button(btn_row, text='Update Curve',
                   command=self._plot_torque_speed).pack(side='left', padx=4, pady=4)
        ttk.Button(btn_row, text='Animate Sweep',
                   command=self._start_animate_ts).pack(side='left', padx=4)
        ttk.Button(btn_row, text='Stop',
                   command=self.sim_running.clear).pack(side='left', padx=4)
        ttk.Button(btn_row, text='Reset',
                   command=self._plot_torque_speed).pack(side='left', padx=4)

        # Results table
        self._ts_text = scrolledtext.ScrolledText(
            frame, height=8, font=('Courier', 10), bg=DARK_BG, fg=CLR_TEXT,
            state='disabled')
        self._ts_text.grid(row=2, column=0, sticky='nsew', padx=4, pady=4)

        self._plot_torque_speed()

    def _plot_torque_speed(self):
        ax = self.ax3
        ax.cla()
        ax.set_facecolor(DARK_AX)

        speeds, T, Ns = self._torque_speed_curve(800)
        ax.plot(speeds, T, color=CLR_TITLE, linewidth=2.5, label='T-N curve')

        # Mark the 6 requested points
        Vth, Rth, Xth, _Ns, ws = self._thevenin()
        X2 = self.var_X2.get()
        R2 = self.var_R2.get()
        Xt = Xth + X2
        mark_spds = [0, 180, 360, 540, 720, 810]
        for spd in mark_spds:
            sv = (Ns - spd) / Ns if spd < Ns else 1.0
            if sv > 0:
                R2s = R2 / sv
                Tm  = (3.0/ws) * Vth**2 * R2s / ((Rth+R2s)**2 + Xt**2)
                ax.plot(spd, Tm, 'o', color=CLR_RED, markersize=8, zorder=5)
                ax.annotate(
                    f'{spd} rpm\n{Tm:.0f} N·m',
                    xy=(spd, Tm),
                    xytext=(spd + Ns*0.04, Tm * 0.88),
                    fontsize=7, color=CLR_ORG,
                    arrowprops=dict(arrowstyle='->', color=CLR_ORG, lw=0.8))

        # LRT annotation
        T_LR = self._torque_at_slip(1.0)
        ax.annotate(
            f'LRT = {T_LR:.0f} N·m',
            xy=(0, T_LR),
            xytext=(Ns * 0.07, T_LR * 0.88),
            fontsize=9, color=CLR_RED,
            arrowprops=dict(arrowstyle='->', color=CLR_RED, lw=1.2))

        # Rated operating point
        _Ns2, Nr2, s_fl, T_r, _ = self._rated_quantities()
        ax.plot(Nr2, T_r, '^', color=CLR_GRN, markersize=11, zorder=6,
                label=f'Rated: {Nr2:.0f} rpm, {T_r:.0f} N·m')

        # Peak torque
        s_pk = R2 / math.sqrt(max(Rth**2 + Xt**2, 1e-12))
        T_pk = self._torque_at_slip(s_pk)
        spd_pk = Ns * (1.0 - s_pk)
        ax.plot(spd_pk, T_pk, 's', color=CLR_ORG, markersize=10, zorder=6,
                label=f'Peak: {T_pk:.0f} N·m @ s={s_pk:.2f}')

        _style_ax(ax,
                  title='Design D Motor — Torque-Speed Characteristic',
                  xlabel='Speed (rpm)', ylabel='Torque (N·m)')
        ax.set_xlim(0, Ns * 1.03)
        ax.set_ylim(0, T_LR * 1.18)
        self.fig3.tight_layout()
        self.canvas3.draw()
        self._update_ts_table()

    def _update_ts_table(self):
        Vth, Rth, Xth, Ns, ws = self._thevenin()
        X2 = self.var_X2.get()
        R2 = self.var_R2.get()
        Xt = Xth + X2

        _Ns, Nr, s_fl, T_r, T_rf = self._rated_quantities()
        T_LR = self._torque_at_slip(1.0)

        txt = self._ts_text
        txt.configure(state='normal')
        txt.delete('1.0', 'end')
        header = (f"  {'Speed':>8}  {'Slip%':>7}  {'T (N·m)':>9}  "
                  f"{'T (ft·lbf)':>11}  {'% rated':>8}\n")
        txt.insert('end', header)
        txt.insert('end', '  ' + '─' * 52 + '\n')
        for spd in [0, 180, 360, 540, 720, 810, int(Nr)]:
            sv = (Ns - spd) / Ns if spd < Ns else 1.0
            if sv <= 0:
                Tv = 0.0
            else:
                R2s = R2 / sv
                Tv  = (3.0/ws) * Vth**2 * R2s / ((Rth+R2s)**2 + Xt**2)
            Tf  = Tv / 1.35582
            pct = Tv / T_r * 100 if T_r > 0 else 0.0
            txt.insert('end',
                f'  {spd:8d}  {sv*100:7.2f}  {Tv:9.1f}  {Tf:11.1f}  {pct:8.1f}\n')
        txt.insert('end', '\n')
        txt.insert('end', f'  T_rated = {T_r:.1f} N·m  ({T_rf:.1f} ft·lbf)\n')
        txt.insert('end', f'  T_LR    = {T_LR:.1f} N·m  '
                          f'({T_LR/1.35582:.1f} ft·lbf)  '
                          f'= {T_LR/T_r*100:.1f}% rated\n')
        txt.configure(state='disabled')

    def _start_animate_ts(self):
        self.sim_running.set()
        th = threading.Thread(target=self._animate_ts_worker, daemon=True)
        th.start()

    def _animate_ts_worker(self):
        speeds, T_full, Ns = self._torque_speed_curve(800)
        idx = 0
        while self.sim_running.is_set() and idx <= len(speeds):
            ax = self.ax3
            ax.cla()
            ax.set_facecolor(DARK_AX)
            ax.plot(speeds[:idx], T_full[:idx], color=CLR_TITLE, linewidth=2)
            ax.set_xlim(0, Ns * 1.03)
            ax.set_ylim(0, max(T_full) * 1.18)
            _style_ax(ax,
                      title='Design D Motor — Torque-Speed (Animated)',
                      xlabel='Speed (rpm)', ylabel='Torque (N·m)', legend=False)
            self.fig3.tight_layout()
            self.canvas3.draw()
            idx += 12
            time.sleep(0.03)
        self.sim_running.clear()
        self._plot_torque_speed()

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 4 — Fault Current
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab4_fault(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='4: Fault Current')
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=3)
        frame.rowconfigure(0, weight=1)

        # Controls
        ctrl = ttk.LabelFrame(frame, text='Fault Parameters')
        ctrl.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        ctrl.columnconfigure(1, weight=1)

        self.var_Zf       = tk.DoubleVar(value=0.05)
        self.var_fault_dur = tk.DoubleVar(value=0.20)
        self.var_XR        = tk.DoubleVar(value=10.0)

        fault_params = [
            ('Fault |Z| (Ω)',   self.var_Zf,        0.01, 1.00),
            ('Duration (s)',    self.var_fault_dur,  0.05, 0.50),
            ('X/R ratio',       self.var_XR,         2.0, 40.0),
        ]
        for ri, (lbl, var, lo, hi) in enumerate(fault_params):
            ttk.Label(ctrl, text=lbl).grid(row=ri, column=0, sticky='w', padx=4, pady=3)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient='horizontal').grid(row=ri, column=1, sticky='ew', padx=4)
            ttk.Label(ctrl, textvariable=var, width=7).grid(row=ri, column=2, padx=2)

        nrow = len(fault_params)
        ttk.Button(ctrl, text='Run Fault Simulation',
                   command=self._run_fault).grid(
            row=nrow, column=0, columnspan=3, pady=6)
        ttk.Button(ctrl, text='Stop', command=self.sim_running.clear).grid(
            row=nrow+1, column=0, columnspan=3, pady=2)

        self._fault_info = ttk.Label(ctrl, text='', justify='left',
                                     font=('Courier', 10), foreground=CLR_ORG)
        self._fault_info.grid(row=nrow+2, column=0, columnspan=3,
                               sticky='w', padx=4, pady=4)

        # Plot
        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.fig4, self.ax4 = plt.subplots(figsize=(8, 5))
        self.fig4.patch.set_facecolor(DARK_BG)
        canvas4 = FigureCanvasTkAgg(self.fig4, master=right)
        canvas4.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self._canvases.append(canvas4)
        self._figures.append(self.fig4)
        self.canvas4 = canvas4
        self._run_fault()

    def _run_fault(self):
        VL   = self.var_VL.get()
        Zf   = max(self.var_Zf.get(), 1e-6)
        dur  = self.var_fault_dur.get()
        XR   = self.var_XR.get()
        freq = self.var_freq.get()

        Vph  = VL / math.sqrt(3.0)
        w    = 2.0 * math.pi * freq
        Rf   = Zf / math.sqrt(1.0 + XR**2)
        Xf   = XR * Rf
        tau  = Xf / (w * Rf) if Rf > 1e-12 else 0.05
        I_pk = Vph * math.sqrt(2.0) / Zf

        t      = np.linspace(0.0, dur, 3000)
        I_sym  = I_pk * np.cos(w * t)
        I_dc   = I_pk * np.exp(-t / tau)
        I_asym = I_sym + I_dc
        I_rms  = I_pk / math.sqrt(2.0)
        I_asym_rms = float(np.sqrt(np.mean(I_asym**2)))

        ax = self.ax4
        ax.cla()
        ax.set_facecolor(DARK_AX)
        ax.plot(t * 1000, I_sym,  color=CLR_TITLE, lw=1.2, label='Symmetrical AC')
        ax.plot(t * 1000, I_asym, color=CLR_RED,   lw=1.4, label='Asymmetrical total')
        ax.plot(t * 1000, I_dc,   color=CLR_ORG,   lw=1.0, ls='--', label='DC offset')
        ax.axhline( I_rms, color=CLR_GRN, lw=1.0, ls=':', label=f'RMS={I_rms:.1f} A')
        ax.axhline(-I_rms, color=CLR_GRN, lw=1.0, ls=':')
        ax.annotate(
            f'X/R={XR:.1f}\nτ={tau*1000:.1f} ms\nI_peak={I_pk:.1f} A\n'
            f'I_rms={I_rms:.1f} A\nI_asym_rms={I_asym_rms:.1f} A',
            xy=(dur*600, I_pk * 0.55), fontsize=8, color=CLR_TEXT,
            bbox=dict(boxstyle='round', facecolor='#313244', alpha=0.85))
        _style_ax(ax, title='Fault Current Simulation',
                  xlabel='Time (ms)', ylabel='Current (A)')
        self.fig4.tight_layout()
        self.canvas4.draw()

        self._fault_info.config(
            text=(f'I_peak       = {I_pk:.2f} A\n'
                  f'I_rms (sym)  = {I_rms:.2f} A\n'
                  f'I_rms (asym) = {I_asym_rms:.2f} A\n'
                  f'X/R          = {XR:.1f}\n'
                  f'τ            = {tau*1000:.2f} ms\n'
                  f'Rf           = {Rf:.5f} Ω'))

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 5 — Protection Coordination
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab5_protection(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='5: Protection')
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=3)
        frame.rowconfigure(0, weight=1)

        ctrl = ttk.LabelFrame(frame, text='Relay Parameters')
        ctrl.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        ctrl.columnconfigure(1, weight=1)

        VL_now  = self.var_VL.get()
        HP_now  = self.var_HP.get()
        I_fl_est = HP_now * 746.0 / (math.sqrt(3.0) * VL_now * 0.90 * 0.90)

        self.var_I_pu    = tk.DoubleVar(value=1.0)
        self.var_TMS     = tk.DoubleVar(value=0.2)
        self.var_CTR     = tk.DoubleVar(value=100.0)
        self.var_I_fault = tk.DoubleVar(value=round(I_fl_est * 8.0, 1))

        prot_params = [
            ('Pickup (× In)',    self.var_I_pu,     0.5,   2.0),
            ('Time Mult (TMS)',  self.var_TMS,      0.05,  1.0),
            ('CT Ratio',        self.var_CTR,       50.0, 400.0),
            ('Fault I (A)',     self.var_I_fault,   I_fl_est, I_fl_est * 20),
        ]
        for ri, (lbl, var, lo, hi) in enumerate(prot_params):
            ttk.Label(ctrl, text=lbl).grid(row=ri, column=0, sticky='w', padx=4, pady=3)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient='horizontal').grid(row=ri, column=1, sticky='ew', padx=4)
            ttk.Label(ctrl, textvariable=var, width=7).grid(row=ri, column=2, padx=2)

        nrow = len(prot_params)
        ttk.Button(ctrl, text='Update TCC Plot',
                   command=self._plot_protection).grid(
            row=nrow, column=0, columnspan=3, pady=8)

        self._prot_info = ttk.Label(ctrl, text='', justify='left',
                                     font=('Courier', 9), foreground=CLR_GRN)
        self._prot_info.grid(row=nrow+1, column=0, columnspan=3,
                              sticky='w', padx=4, pady=2)

        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.fig5, self.ax5 = plt.subplots(figsize=(8, 5))
        self.fig5.patch.set_facecolor(DARK_BG)
        canvas5 = FigureCanvasTkAgg(self.fig5, master=right)
        canvas5.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self._canvases.append(canvas5)
        self._figures.append(self.fig5)
        self.canvas5 = canvas5
        self._plot_protection()

    def _idmt_time(self, I_mult, TMS, curve='SI'):
        if I_mult <= 1.0:
            return 1e6
        if curve == 'SI':
            return TMS * 0.14 / (I_mult**0.02 - 1.0)
        if curve == 'VI':
            return TMS * 13.5 / (I_mult - 1.0)
        if curve == 'EI':
            return TMS * 80.0 / (I_mult**2 - 1.0)
        return 1e6

    def _plot_protection(self):
        VL   = self.var_VL.get()
        HP   = self.var_HP.get()
        I_fl = HP * 746.0 / (math.sqrt(3.0) * VL * 0.90 * 0.90)
        Ip   = self.var_I_pu.get() * I_fl
        TMS  = self.var_TMS.get()
        I_f  = self.var_I_fault.get()

        I_lo = max(Ip * 1.01, 1.0)
        I_hi = max(I_f * 2.5, I_lo * 10)
        I_range = np.logspace(math.log10(I_lo), math.log10(I_hi), 600)

        t_SI  = np.array([self._idmt_time(i/Ip, TMS, 'SI')  for i in I_range])
        t_VI  = np.array([self._idmt_time(i/Ip, TMS, 'VI')  for i in I_range])
        t_EI  = np.array([self._idmt_time(i/Ip, TMS, 'EI')  for i in I_range])
        t_fuse = np.clip((I_fl * 6.0)**2 * 0.015 / I_range**2, 1e-3, 1e4)
        t_ol   = np.clip(8.0 * I_fl**2 / np.maximum((I_range - I_fl), 0.01)**2,
                         1e-3, 300.0)

        ax = self.ax5
        ax.cla()
        ax.set_facecolor(DARK_AX)
        ax.loglog(I_range, t_SI,  color=CLR_TITLE, lw=2,   label='IDMT SI')
        ax.loglog(I_range, t_VI,  color=CLR_GRN,   lw=2,   label='IDMT VI')
        ax.loglog(I_range, t_EI,  color=CLR_ORG,   lw=2,   label='IDMT EI')
        ax.loglog(I_range, t_fuse,color=CLR_RED,   lw=2,   label='HRC Fuse')
        ax.loglog(I_range, t_ol,  color=CLR_PUR,   lw=1.5, ls='--',
                  label='Overload Relay')

        ax.axvline(I_f,  color='white',   ls=':', lw=1.2,
                   label=f'I_fault={I_f:.0f} A')
        ax.axvline(I_fl, color='#6c7086', ls='--', lw=1.0,
                   label=f'I_fl={I_fl:.1f} A')

        t_op_SI = self._idmt_time(I_f / Ip, TMS, 'SI')
        t_op_VI = self._idmt_time(I_f / Ip, TMS, 'VI')
        t_op_EI = self._idmt_time(I_f / Ip, TMS, 'EI')

        if t_op_SI < 100:
            ax.annotate(f'SI={t_op_SI:.3f}s',
                        xy=(I_f, t_op_SI),
                        xytext=(I_f * 1.5, t_op_SI * 3),
                        color=CLR_TITLE, fontsize=8,
                        arrowprops=dict(arrowstyle='->', color=CLR_TITLE))

        _style_ax(ax, title='Protection Coordination — Time-Current Curves (TCC)',
                  xlabel='Current (A)', ylabel='Operating Time (s)')
        ax.set_ylim(0.001, 1000)
        self.fig5.tight_layout()
        self.canvas5.draw()

        self._prot_info.config(
            text=(f'I_fl     = {I_fl:.2f} A\n'
                  f'I_pickup = {Ip:.2f} A\n'
                  f't_SI     = {t_op_SI:.4f} s\n'
                  f't_VI     = {t_op_VI:.4f} s\n'
                  f't_EI     = {t_op_EI:.4f} s'))

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 6 — Speed Controller
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab6_speed_ctrl(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='6: Speed Ctrl')
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=3)
        frame.rowconfigure(0, weight=1)

        ctrl = ttk.LabelFrame(frame, text='Controller Parameters')
        ctrl.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        ctrl.columnconfigure(1, weight=1)

        self.var_Kp         = tk.DoubleVar(value=6.0)
        self.var_Ki         = tk.DoubleVar(value=0.8)
        self.var_Kd         = tk.DoubleVar(value=0.05)
        self.var_load_step  = tk.DoubleVar(value=120.0)
        self.var_ctrl_type  = tk.StringVar(value='PID')
        self.var_spd_ref_pct = tk.DoubleVar(value=95.0)

        sc_params = [
            ('Kp',                   self.var_Kp,           0.1, 30.0),
            ('Ki',                   self.var_Ki,           0.0,  5.0),
            ('Kd',                   self.var_Kd,           0.0,  2.0),
            ('Load Step (N·m)',      self.var_load_step,    0.0, 500.0),
            ('Speed Ref (% of Ns)', self.var_spd_ref_pct, 50.0, 100.0),
        ]
        for ri, (lbl, var, lo, hi) in enumerate(sc_params):
            ttk.Label(ctrl, text=lbl).grid(row=ri, column=0, sticky='w', padx=4, pady=2)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient='horizontal').grid(row=ri, column=1, sticky='ew', padx=4)
            ttk.Label(ctrl, textvariable=var, width=6).grid(row=ri, column=2)

        nrow = len(sc_params)
        ttk.Label(ctrl, text='Controller type').grid(
            row=nrow, column=0, sticky='w', padx=4, pady=4)
        for ci, ctype in enumerate(['PID', 'Fuzzy']):
            ttk.Radiobutton(ctrl, text=ctype, variable=self.var_ctrl_type,
                            value=ctype).grid(row=nrow+ci, column=1, sticky='w')

        ttk.Button(ctrl, text='Start Simulation',
                   command=self._start_speed_ctrl).grid(
            row=nrow+2, column=0, columnspan=3, pady=4)
        ttk.Button(ctrl, text='Stop',
                   command=self.sim_running.clear).grid(
            row=nrow+3, column=0, columnspan=3, pady=2)
        ttk.Button(ctrl, text='Reset / Clear',
                   command=self._reset_speed_ctrl_plot).grid(
            row=nrow+4, column=0, columnspan=3, pady=2)

        # Plot
        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.fig6, (self.ax6a, self.ax6b) = plt.subplots(2, 1, figsize=(8, 6),
                                                           sharex=True)
        self.fig6.patch.set_facecolor(DARK_BG)
        canvas6 = FigureCanvasTkAgg(self.fig6, master=right)
        canvas6.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self._canvases.append(canvas6)
        self._figures.append(self.fig6)
        self.canvas6 = canvas6
        self._reset_speed_ctrl_plot()

    def _reset_speed_ctrl_plot(self):
        for ax in (self.ax6a, self.ax6b):
            ax.cla()
            ax.set_facecolor(DARK_AX)
            ax.tick_params(colors=CLR_TEXT)
            ax.grid(True, color=CLR_GRID, linestyle='--', alpha=0.7)
            ax.spines[:].set_color(CLR_SPINE)
        self.ax6a.set_title('Speed Controller Simulation', color=CLR_TITLE, fontsize=12)
        self.ax6a.set_ylabel('Speed (rpm)', color=CLR_TEXT)
        self.ax6b.set_ylabel('Torque (N·m)', color=CLR_TEXT)
        self.ax6b.set_xlabel('Time (s)', color=CLR_TEXT)
        self.fig6.tight_layout()
        self.canvas6.draw()

    def _start_speed_ctrl(self):
        self.sim_running.set()
        th = threading.Thread(target=self._speed_ctrl_worker, daemon=True)
        th.start()

    def _speed_ctrl_worker(self):
        freq  = self.var_freq.get()
        poles = self.var_poles.get()
        Ns    = 120.0 * freq / poles
        ws    = 2.0 * math.pi * Ns / 60.0

        _, _, _, J_SI = self._flywheel_calc()
        HP_now  = self.var_HP.get()
        J_motor = HP_now * HP_INERTIA_COEFF   # kg·m² — empirical estimate
        J_tot   = max(J_SI + J_motor, 0.1)

        Kp   = self.var_Kp.get()
        Ki   = self.var_Ki.get()
        Kd   = self.var_Kd.get()
        T_step = self.var_load_step.get()
        ref_pct = self.var_spd_ref_pct.get() / 100.0
        omega_ref = ws * ref_pct
        ctrl_type = self.var_ctrl_type.get()

        dt      = 0.02
        t_end   = 10.0
        t_arr   = np.arange(0.0, t_end, dt)
        omega   = 0.0
        integral = 0.0
        prev_err = 0.0

        t_hist   = []
        spd_hist = []
        T_hist   = []

        for ti in t_arr:
            if not self.sim_running.is_set():
                break

            T_load = T_step if ti >= 2.0 else 0.0
            err    = omega_ref - omega

            if ctrl_type == 'PID':
                integral += err * dt
                integral  = max(min(integral, 5000.0), -5000.0)
                deriv = (err - prev_err) / dt if dt > 0 else 0.0
                u     = Kp * err + Ki * integral + Kd * deriv
                u     = max(min(u, 2000.0), -2000.0)
                prev_err = err
            else:
                u = self._fuzzy_ctrl(err, omega_ref)

            # Map u → slip: normalise by (ws × CTRL_U_MAX) so full u gives s≈1
            slip_ctrl = max(min(abs(u) / (ws * CTRL_U_MAX), 0.999), 1e-4)
            T_em  = self._torque_at_slip(slip_ctrl)
            T_net = T_em - T_load
            alpha = T_net / J_tot
            omega += alpha * dt
            omega  = max(0.0, min(omega, ws * 1.1))

            t_hist.append(ti)
            spd_hist.append(omega * 60.0 / (2.0 * math.pi))
            T_hist.append(T_em)

            if len(t_hist) % 4 == 0:
                self._draw_speed_ctrl(t_hist, spd_hist, T_hist,
                                       Ns * ref_pct, Ns)
            time.sleep(0.001)

        self._draw_speed_ctrl(t_hist, spd_hist, T_hist, Ns * ref_pct, Ns)
        self.sim_running.clear()

    def _fuzzy_ctrl(self, err, omega_ref):
        e_n = err / max(omega_ref, 1e-3)
        if   e_n >  0.20: return  1500.0
        elif e_n >  0.08: return   600.0
        elif e_n > -0.08: return    60.0
        elif e_n > -0.20: return  -300.0
        else:             return  -800.0

    def _draw_speed_ctrl(self, t, spd, T, ref_rpm, Ns):
        for ax in (self.ax6a, self.ax6b):
            ax.cla()
            ax.set_facecolor(DARK_AX)
            ax.tick_params(colors=CLR_TEXT)
            ax.grid(True, color=CLR_GRID, linestyle='--', alpha=0.7)
            ax.spines[:].set_color(CLR_SPINE)

        self.ax6a.plot(t, spd, color=CLR_TITLE, lw=1.5, label='Speed')
        self.ax6a.axhline(ref_rpm, color=CLR_RED, ls='--', lw=1.2, label='Reference')
        self.ax6a.set_ylabel('Speed (rpm)', color=CLR_TEXT)
        self.ax6a.set_title('Speed Controller Simulation', color=CLR_TITLE, fontsize=12)
        self.ax6a.legend(facecolor='#313244', edgecolor=CLR_SPINE,
                          labelcolor=CLR_TEXT, fontsize=8)

        self.ax6b.plot(t, T, color=CLR_GRN, lw=1.5, label='T_em')
        self.ax6b.set_ylabel('Torque (N·m)', color=CLR_TEXT)
        self.ax6b.set_xlabel('Time (s)', color=CLR_TEXT)
        self.ax6b.legend(facecolor='#313244', edgecolor=CLR_SPINE,
                          labelcolor=CLR_TEXT, fontsize=8)

        self.fig6.tight_layout()
        self.canvas6.draw()

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 7 — Thermal Analysis
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab7_thermal(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='7: Thermal')
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=3)
        frame.rowconfigure(0, weight=1)

        ctrl = ttk.LabelFrame(frame, text='Thermal Parameters')
        ctrl.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        ctrl.columnconfigure(1, weight=1)

        self.var_theta_amb = tk.DoubleVar(value=40.0)
        self.var_load_fac  = tk.DoubleVar(value=1.0)
        self.var_tau_th    = tk.DoubleVar(value=45.0)    # minutes
        self.var_eta_th    = tk.DoubleVar(value=0.90)    # efficiency

        th_params = [
            ('Ambient Temp (°C)',   self.var_theta_amb,  0.0,  50.0),
            ('Load Factor',        self.var_load_fac,   0.1,   1.5),
            ('Thermal τ (min)',    self.var_tau_th,      5.0, 120.0),
            ('Motor Efficiency',   self.var_eta_th,     0.75,  0.97),
        ]
        for ri, (lbl, var, lo, hi) in enumerate(th_params):
            ttk.Label(ctrl, text=lbl).grid(row=ri, column=0, sticky='w', padx=4, pady=3)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient='horizontal').grid(row=ri, column=1, sticky='ew', padx=4)
            ttk.Label(ctrl, textvariable=var, width=6).grid(row=ri, column=2, padx=2)

        nrow = len(th_params)
        ttk.Button(ctrl, text='Run Thermal Simulation',
                   command=self._start_thermal).grid(
            row=nrow, column=0, columnspan=3, pady=6)
        ttk.Button(ctrl, text='Stop', command=self.sim_running.clear).grid(
            row=nrow+1, column=0, columnspan=3, pady=2)
        ttk.Button(ctrl, text='Reset', command=self._reset_thermal_plots).grid(
            row=nrow+2, column=0, columnspan=3, pady=2)

        self._thermal_alarm = ttk.Label(ctrl, text='', font=('Courier', 12, 'bold'))
        self._thermal_alarm.grid(row=nrow+3, column=0, columnspan=3, pady=6)

        self._thermal_info = ttk.Label(ctrl, text='', justify='left',
                                        font=('Courier', 9), foreground=CLR_GRN)
        self._thermal_info.grid(row=nrow+4, column=0, columnspan=3,
                                 sticky='w', padx=4)

        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self.fig7a, self.ax7a = plt.subplots(figsize=(8, 3))
        self.fig7a.patch.set_facecolor(DARK_BG)
        canvas7a = FigureCanvasTkAgg(self.fig7a, master=right)
        canvas7a.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self._canvases.append(canvas7a)
        self._figures.append(self.fig7a)
        self.canvas7a = canvas7a

        self.fig7b, self.ax7b = plt.subplots(figsize=(8, 3))
        self.fig7b.patch.set_facecolor(DARK_BG)
        canvas7b = FigureCanvasTkAgg(self.fig7b, master=right)
        canvas7b.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        self._canvases.append(canvas7b)
        self._figures.append(self.fig7b)
        self.canvas7b = canvas7b
        self._reset_thermal_plots()

    def _reset_thermal_plots(self):
        for ax, fig, canvas in [(self.ax7a, self.fig7a, self.canvas7a),
                                  (self.ax7b, self.fig7b, self.canvas7b)]:
            ax.cla()
            ax.set_facecolor(DARK_AX)
            ax.tick_params(colors=CLR_TEXT)
            ax.grid(True, color=CLR_GRID, ls='--', alpha=0.7)
            ax.spines[:].set_color(CLR_SPINE)
            fig.tight_layout()
            canvas.draw()

    def _start_thermal(self):
        self.sim_running.set()
        th = threading.Thread(target=self._thermal_worker, daemon=True)
        th.start()

    def _thermal_worker(self):
        HP     = self.var_HP.get()
        LF     = self.var_load_fac.get()
        eta    = self.var_eta_th.get()
        P_out  = HP * 746.0 * LF
        P_in   = P_out / max(eta, 0.01)
        P_loss = P_in - P_out

        # Loss breakdown fractions
        P_sc   = P_loss * 0.40
        P_rc   = P_loss * 0.35
        P_core = P_loss * 0.15
        P_fric = P_loss * 0.10

        theta_amb = self.var_theta_amb.get()
        tau_min   = self.var_tau_th.get()
        tau_s     = tau_min * 60.0
        k_th      = P_loss / SS_TEMP_RISE_DEG if P_loss > 0 else 1.0   # W/°C
        C_th      = k_th * tau_s

        theta_ss  = theta_amb + P_loss / k_th

        dt    = 15.0       # seconds per step
        t_end = tau_s * 4.0
        t_arr = np.arange(0.0, t_end + dt, dt)
        theta = theta_amb

        t_hist  = []
        th_hist = []
        for ti in t_arr:
            if not self.sim_running.is_set():
                break
            dth    = (P_loss - k_th * (theta - theta_amb)) / C_th * dt
            theta += dth
            t_hist.append(ti / 60.0)
            th_hist.append(theta)

        # Plot temperature rise
        ax = self.ax7a
        ax.cla()
        ax.set_facecolor(DARK_AX)
        ax.plot(t_hist, th_hist, color=CLR_TITLE, lw=2.0, label='θ(t)')
        ax.axhline(155.0, color=CLR_RED, ls='--', lw=1.5,
                   label='Class F limit 155°C')
        ax.axhline(130.0, color=CLR_ORG, ls=':', lw=1.2,
                   label='Class B limit 130°C')
        ax.axhline(theta_ss, color=CLR_GRN, ls=':', lw=1.0,
                   label=f'θ_ss = {theta_ss:.1f}°C')
        ax.fill_between(t_hist, th_hist, theta_amb, alpha=0.15, color=CLR_TITLE)
        _style_ax(ax, title='Motor Temperature Rise', xlabel='Time (min)',
                  ylabel='Temperature (°C)')
        self.fig7a.tight_layout()
        self.canvas7a.draw()

        # Pie chart for loss breakdown
        ax2 = self.ax7b
        ax2.cla()
        ax2.set_facecolor(DARK_AX)
        pie_labels = ['Stator Cu', 'Rotor Cu', 'Core', 'Friction']
        pie_sizes  = [P_sc, P_rc, P_core, P_fric]
        pie_colors = [CLR_TITLE, CLR_RED, CLR_GRN, CLR_ORG]
        _w, _texts, autotxts = ax2.pie(
            pie_sizes, labels=pie_labels, colors=pie_colors,
            autopct='%1.1f%%', textprops=dict(color=CLR_TEXT))
        for at in autotxts:
            at.set_color(DARK_BG)
        ax2.set_title(f'Loss Breakdown — Total {P_loss:.0f} W  (LF={LF:.2f})',
                       color=CLR_TITLE, fontsize=11)
        self.fig7b.tight_layout()
        self.canvas7b.draw()

        max_th = max(th_hist) if th_hist else theta_amb
        if max_th > 155:
            self._thermal_alarm.config(text='⚠  OVER-TEMPERATURE!', foreground=CLR_RED)
        else:
            self._thermal_alarm.config(text='✓  Temperature within Class F',
                                        foreground=CLR_GRN)

        self._thermal_info.config(
            text=(f'P_loss  = {P_loss:.0f} W\n'
                  f'θ_ss    = {theta_ss:.1f} °C\n'
                  f'θ_max   = {max_th:.1f} °C\n'
                  f'τ_th    = {tau_min:.0f} min'))
        self.sim_running.clear()

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 8 — Economic Analysis
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab8_economic(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='8: Economics')
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=3)
        frame.rowconfigure(0, weight=1)

        ctrl = ttk.LabelFrame(frame, text='Economic Parameters')
        ctrl.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        ctrl.columnconfigure(1, weight=1)

        self.var_ecost     = tk.DoubleVar(value=0.12)
        self.var_hrs_yr    = tk.DoubleVar(value=4000.0)
        self.var_lf_ec     = tk.DoubleVar(value=0.85)
        self.var_eta_std   = tk.DoubleVar(value=0.90)
        self.var_eta_prem  = tk.DoubleVar(value=0.935)
        self.var_prem_cost = tk.DoubleVar(value=3000.0)
        self.var_co2_fac   = tk.DoubleVar(value=0.45)

        ec_params = [
            ('Energy cost ($/kWh)',  self.var_ecost,     0.05, 0.50),
            ('Hours / year',        self.var_hrs_yr,   1000, 8760),
            ('Load factor',         self.var_lf_ec,     0.30,  1.00),
            ('Std efficiency',      self.var_eta_std,   0.80,  0.96),
            ('Prem efficiency',     self.var_eta_prem,  0.85,  0.97),
            ('Prem motor cost ($)', self.var_prem_cost, 500, 10000),
            ('CO₂ factor (kg/kWh)', self.var_co2_fac,   0.10,  1.00),
        ]
        for ri, (lbl, var, lo, hi) in enumerate(ec_params):
            ttk.Label(ctrl, text=lbl).grid(row=ri, column=0, sticky='w', padx=4, pady=2)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient='horizontal').grid(row=ri, column=1, sticky='ew', padx=4)
            ttk.Label(ctrl, textvariable=var, width=7).grid(row=ri, column=2, padx=2)

        nrow = len(ec_params)
        ttk.Button(ctrl, text='Calculate',
                   command=self._run_economics).grid(
            row=nrow, column=0, columnspan=3, pady=8)

        self._econ_info = ttk.Label(ctrl, text='', justify='left',
                                     font=('Courier', 9), foreground=CLR_GRN)
        self._econ_info.grid(row=nrow+1, column=0, columnspan=3,
                              sticky='w', padx=4)

        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self.fig8a, self.ax8a = plt.subplots(figsize=(8, 3))
        self.fig8a.patch.set_facecolor(DARK_BG)
        canvas8a = FigureCanvasTkAgg(self.fig8a, master=right)
        canvas8a.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self._canvases.append(canvas8a)
        self._figures.append(self.fig8a)
        self.canvas8a = canvas8a

        self.fig8b, self.ax8b = plt.subplots(figsize=(8, 3))
        self.fig8b.patch.set_facecolor(DARK_BG)
        canvas8b = FigureCanvasTkAgg(self.fig8b, master=right)
        canvas8b.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        self._canvases.append(canvas8b)
        self._figures.append(self.fig8b)
        self.canvas8b = canvas8b
        self._run_economics()

    def _run_economics(self):
        HP        = self.var_HP.get()
        cost_kwh  = self.var_ecost.get()
        hrs       = self.var_hrs_yr.get()
        LF        = self.var_lf_ec.get()
        eta_std   = max(self.var_eta_std.get(),  0.01)
        eta_prem  = max(self.var_eta_prem.get(), 0.01)
        prem_cost = self.var_prem_cost.get()
        co2_fac   = self.var_co2_fac.get()

        P_out  = HP * 746.0 * LF / 1000.0   # kW
        P_std  = P_out / eta_std
        P_prem = P_out / eta_prem
        E_std  = P_std  * hrs
        E_prem = P_prem * hrs
        c_std  = E_std  * cost_kwh
        c_prem = E_prem * cost_kwh
        savings = c_std - c_prem
        payback = prem_cost / savings if savings > 1e-6 else float('inf')
        co2_std  = E_std  * co2_fac / 1000.0
        co2_prem = E_prem * co2_fac / 1000.0
        co2_save = co2_std - co2_prem

        # Bar chart
        ax = self.ax8a
        ax.cla()
        ax.set_facecolor(DARK_AX)
        cats   = ['Energy\n(MWh/yr)', 'Cost\n($k/yr)', 'CO₂\n(t/yr)']
        v_std  = [E_std / 1000.0, c_std / 1000.0, co2_std]
        v_prem = [E_prem / 1000.0, c_prem / 1000.0, co2_prem]
        x  = np.arange(len(cats))
        w  = 0.35
        ax.bar(x - w/2, v_std,  w, label='Standard',   color=CLR_RED,   alpha=0.85)
        ax.bar(x + w/2, v_prem, w, label='Premium IE', color=CLR_GRN,   alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(cats, color=CLR_TEXT, fontsize=9)
        _style_ax(ax, title='Annual Operating Cost Comparison', ylabel='Value')
        self.fig8a.tight_layout()
        self.canvas8a.draw()

        # Payback chart
        ax2 = self.ax8b
        ax2.cla()
        ax2.set_facecolor(DARK_AX)
        yrs  = np.linspace(0, 15, 300)
        cum  = savings * yrs - prem_cost
        ax2.plot(yrs, cum, color=CLR_TITLE, lw=2.0, label='Cumulative savings')
        ax2.axhline(0, color=CLR_RED, ls='--', lw=1.2, label='Break-even')
        ax2.fill_between(yrs, cum, 0, where=(cum >= 0), alpha=0.18, color=CLR_GRN)
        ax2.fill_between(yrs, cum, 0, where=(cum < 0),  alpha=0.18, color=CLR_RED)
        if savings > 0:
            ax2.axvline(payback, color=CLR_ORG, ls=':', lw=1.2,
                        label=f'Payback={payback:.1f} yr')
        _style_ax(ax2, title='Payback Analysis',
                  xlabel='Years', ylabel='Cumulative Savings ($)')
        self.fig8b.tight_layout()
        self.canvas8b.draw()

        self._econ_info.config(
            text=(f'P_in (std)  = {P_std:.2f} kW\n'
                  f'P_in (prem) = {P_prem:.2f} kW\n'
                  f'Savings     = ${savings:.0f}/yr\n'
                  f'Payback     = {payback:.2f} yr\n'
                  f'CO₂ saved   = {co2_save:.2f} t/yr'))

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 9 — Harmonic & Power Quality
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab9_harmonics(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='9: Harmonics')
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=3)
        frame.rowconfigure(0, weight=1)

        ctrl = ttk.LabelFrame(frame, text='Harmonic Content (% of fundamental)')
        ctrl.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        ctrl.columnconfigure(1, weight=1)

        self.var_h5  = tk.DoubleVar(value=6.0)
        self.var_h7  = tk.DoubleVar(value=4.0)
        self.var_h11 = tk.DoubleVar(value=2.0)
        self.var_h13 = tk.DoubleVar(value=1.5)
        self.var_Zs  = tk.DoubleVar(value=0.05)   # source impedance (pu)

        harm_params = [
            ('5th harmonic (%)',  self.var_h5,  0.0, 25.0),
            ('7th harmonic (%)',  self.var_h7,  0.0, 20.0),
            ('11th harmonic (%)', self.var_h11, 0.0, 10.0),
            ('13th harmonic (%)', self.var_h13, 0.0, 10.0),
            ('Source Zs (pu)',    self.var_Zs,  0.01, 0.20),
        ]
        for ri, (lbl, var, lo, hi) in enumerate(harm_params):
            ttk.Label(ctrl, text=lbl).grid(row=ri, column=0, sticky='w', padx=4, pady=3)
            ttk.Scale(ctrl, from_=lo, to=hi, variable=var,
                      orient='horizontal').grid(row=ri, column=1, sticky='ew', padx=4)
            ttk.Label(ctrl, textvariable=var, width=6).grid(row=ri, column=2, padx=2)

        nrow = len(harm_params)
        ttk.Button(ctrl, text='Update Plots',
                   command=self._run_harmonics).grid(
            row=nrow, column=0, columnspan=3, pady=8)

        self._harm_info = ttk.Label(ctrl, text='', justify='left',
                                     font=('Courier', 9), foreground=CLR_GRN)
        self._harm_info.grid(row=nrow+1, column=0, columnspan=3,
                              sticky='w', padx=4)

        right = ttk.Frame(frame)
        right.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self.fig9a, self.ax9a = plt.subplots(figsize=(8, 3))
        self.fig9a.patch.set_facecolor(DARK_BG)
        canvas9a = FigureCanvasTkAgg(self.fig9a, master=right)
        canvas9a.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self._canvases.append(canvas9a)
        self._figures.append(self.fig9a)
        self.canvas9a = canvas9a

        self.fig9b, self.ax9b = plt.subplots(figsize=(8, 3))
        self.fig9b.patch.set_facecolor(DARK_BG)
        canvas9b = FigureCanvasTkAgg(self.fig9b, master=right)
        canvas9b.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        self._canvases.append(canvas9b)
        self._figures.append(self.fig9b)
        self.canvas9b = canvas9b
        self._run_harmonics()

    def _run_harmonics(self):
        freq = self.var_freq.get()
        h5   = self.var_h5.get()  / 100.0
        h7   = self.var_h7.get()  / 100.0
        h11  = self.var_h11.get() / 100.0
        h13  = self.var_h13.get() / 100.0
        Zs   = self.var_Zs.get()
        w    = 2.0 * math.pi * freq
        t    = np.linspace(0.0, 2.0 / freq, 1200)

        I1     = 1.0   # pu
        I_wave = (I1 * np.sin(w * t)
                  + h5  * I1 * np.sin(5.0  * w * t)
                  + h7  * I1 * np.sin(7.0  * w * t)
                  + h11 * I1 * np.sin(11.0 * w * t)
                  + h13 * I1 * np.sin(13.0 * w * t))

        THD_I = math.sqrt(h5**2 + h7**2 + h11**2 + h13**2) * 100.0
        THD_V = THD_I * Zs
        DPF   = 0.85
        TPF   = DPF / math.sqrt(1.0 + (THD_I / 100.0)**2)

        # Waveform
        ax = self.ax9a
        ax.cla()
        ax.set_facecolor(DARK_AX)
        ax.plot(t * 1000.0, I_wave,           color=CLR_TITLE, lw=1.5, label='Total')
        ax.plot(t * 1000.0, I1 * np.sin(w*t), color=CLR_GRN,   lw=1.0,
                ls='--', label='Fundamental')
        _style_ax(ax, title='Motor Phase Current Waveform',
                  xlabel='Time (ms)', ylabel='Current (pu)')
        self.fig9a.tight_layout()
        self.canvas9a.draw()

        # Spectrum
        ax2 = self.ax9b
        ax2.cla()
        ax2.set_facecolor(DARK_AX)
        h_orders  = [1, 5, 7, 11, 13]
        h_mags    = [100.0, h5*100, h7*100, h11*100, h13*100]
        h_labels  = ['1st', '5th', '7th', '11th', '13th']
        h_colors  = [CLR_TITLE, CLR_RED, CLR_ORG, CLR_GRN, CLR_PUR]
        bars = ax2.bar(h_labels, h_mags, color=h_colors, edgecolor=CLR_SPINE, width=0.5)
        for bar, mag in zip(bars, h_mags):
            ax2.text(bar.get_x() + bar.get_width()/2.0, mag + 0.3,
                     f'{mag:.1f}%', ha='center', va='bottom',
                     color=CLR_TEXT, fontsize=8)

        # IEEE 519 limits
        ax2.axhline(4.0, color=CLR_RED, ls='--', lw=1.2,
                    label='IEEE 519 5th/7th limit (~4%)')
        ax2.axhline(2.0, color=CLR_ORG, ls=':', lw=1.0,
                    label='IEEE 519 11th/13th limit (~2%)')
        _style_ax(ax2,
                  title=(f'Harmonic Spectrum   THD_I={THD_I:.1f}%   '
                         f'THD_V={THD_V:.1f}%   TPF={TPF:.3f}'),
                  xlabel='Harmonic Order', ylabel='Magnitude (% fundamental)')
        self.fig9b.tight_layout()
        self.canvas9b.draw()

        self._harm_info.config(
            text=(f'THD_I   = {THD_I:.2f} %\n'
                  f'THD_V   = {THD_V:.2f} %\n'
                  f'DPF     = {DPF:.3f}\n'
                  f'TPF     = {TPF:.3f}\n'
                  f'IEEE519: 5th ≤ 4%, 11th ≤ 2%'))

    # ══════════════════════════════════════════════════════════════════════════
    #  TAB 10 — Comprehensive Analysis
    # ══════════════════════════════════════════════════════════════════════════
    def _build_tab10_comprehensive(self):
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text='10: Comprehensive')
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=2)
        frame.rowconfigure(0, weight=1)

        # Left: scrollable summary
        left = ttk.LabelFrame(frame, text='Full Summary')
        left.grid(row=0, column=0, sticky='nsew', padx=6, pady=6)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        btn_bar = ttk.Frame(left)
        btn_bar.grid(row=0, column=0, sticky='ew', pady=4)
        ttk.Button(btn_bar, text='Run All Analyses',
                   command=self._run_all_analyses).pack(side='left', padx=4)
        ttk.Button(btn_bar, text='Export to Console',
                   command=self._export_summary).pack(side='left', padx=4)

        self._comp_text = scrolledtext.ScrolledText(
            left, font=('Courier', 10), bg=DARK_BG, fg=CLR_TEXT,
            insertbackground='white', state='disabled')
        self._comp_text.grid(row=1, column=0, sticky='nsew', padx=4, pady=4)

        # Right: sensitivity analysis (2×2 subplots)
        right = ttk.LabelFrame(frame, text='Sensitivity Analysis (±20%)')
        right.grid(row=0, column=1, sticky='nsew', padx=6, pady=6)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.fig10, self.axes10 = plt.subplots(2, 2, figsize=(8, 6))
        self.fig10.patch.set_facecolor(DARK_BG)
        for ax in self.axes10.flat:
            ax.set_facecolor(DARK_AX)
        canvas10 = FigureCanvasTkAgg(self.fig10, master=right)
        canvas10.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        self._canvases.append(canvas10)
        self._figures.append(self.fig10)
        self.canvas10 = canvas10

        self._run_all_analyses()

    def _run_all_analyses(self):
        self._update_comprehensive_text()
        self._plot_sensitivity()
        self._update_input_results()
        self._plot_torque_speed()
        self._run_fault()
        self._plot_protection()
        self._run_economics()
        self._run_harmonics()

    def _update_comprehensive_text(self):
        try:
            Ns, Nr, s_fl, T_Nm, T_ft = self._rated_quantities()
            r_in, weight_lb, WK2, J_SI = self._flywheel_calc()
            Vth, Rth, Xth, _Ns, ws = self._thevenin()
            X2   = self.var_X2.get()
            R2   = self.var_R2.get()
            Xt   = Xth + X2
            T_LR = self._torque_at_slip(1.0)
            s_pk = R2 / math.sqrt(max(Rth**2 + Xt**2, 1e-12))
            T_pk = self._torque_at_slip(s_pk)

            VL   = self.var_VL.get()
            HP   = self.var_HP.get()
            I_fl = HP * 746.0 / (math.sqrt(3.0) * VL * 0.90 * 0.90)

            lines = [
                '═' * 56,
                '  DESIGN D MOTOR — COMPREHENSIVE SUMMARY',
                '═' * 56,
                '',
                '  MOTOR NAMEPLATE & PARAMETERS',
                '  ' + '─' * 40,
                f'  Rating : {HP:.0f} HP,  {VL:.0f} V,  '
                f'{self.var_freq.get():.0f} Hz,  {self.var_poles.get()} poles',
                f'  R1={self.var_R1.get():.3f} Ω   X1={self.var_X1.get():.3f} Ω',
                f'  R2={self.var_R2.get():.3f} Ω   X2={self.var_X2.get():.3f} Ω',
                f'  Xm={self.var_Xm.get():.2f} Ω',
                '',
                '  SYNCHRONOUS & RATED QUANTITIES',
                '  ' + '─' * 40,
                f'  Ns        = {Ns:.1f} rpm',
                f'  Nr        = {Nr:.1f} rpm',
                f'  s_fl      = {s_fl*100:.2f} %',
                f'  T_rated   = {T_Nm:.2f} N·m   ({T_ft:.2f} ft·lbf)',
                f'  I_fl(est) = {I_fl:.2f} A',
                '',
                '  DESIGN D CHARACTERISTICS',
                '  ' + '─' * 40,
                f'  T_LR      = {T_LR:.1f} N·m   ({T_LR/1.35582:.1f} ft·lbf)',
                f'  T_LR/T_r  = {T_LR/T_Nm*100:.1f} %  (should be ~275%)',
                f'  T_peak    = {T_pk:.1f} N·m  at s={s_pk:.3f}  '
                f'({Ns*(1-s_pk):.0f} rpm)',
                '',
                '  THEVENIN EQUIVALENT CIRCUIT',
                '  ' + '─' * 40,
                f'  Vth = {Vth:.3f} V',
                f'  Rth = {Rth:.5f} Ω',
                f'  Xth = {Xth:.5f} Ω',
                f'  Xt  = Xth+X2 = {Xt:.4f} Ω',
                f'  ws  = {ws:.3f} rad/s',
                '',
                '  FLYWHEEL LOAD',
                '  ' + '─' * 40,
                f'  Diameter = {self.var_fw_diam.get():.3f} in',
                f'  Radius r = {r_in:.4f} in  = {r_in/12:.4f} ft',
                f'  Weight W = {weight_lb:.1f} lb',
                f'  WK²      = {WK2:.1f} lb·ft²',
                f'  J  (SI)  = {J_SI:.3f} kg·m²',
                '',
                '  TORQUES AT SELECTED SPEEDS',
                '  ' + '─' * 40,
                f"  {'Speed':>8}  {'Slip%':>7}  {'T(N·m)':>9}  "
                f"{'T(ftlbf)':>10}  {'%rated':>7}",
                '  ' + '─' * 50,
            ]
            for spd in [0, 180, 360, 540, 720, 810, int(Nr)]:
                sv = (Ns - spd) / Ns if spd < Ns else 1.0
                Tv = 0.0 if sv <= 0 else self._torque_at_slip(sv)
                Tf = Tv / 1.35582
                pct = Tv / T_Nm * 100 if T_Nm > 0 else 0.0
                lines.append(
                    f'  {spd:8d}  {sv*100:7.2f}  {Tv:9.1f}  {Tf:10.1f}  {pct:7.1f}')
            lines.append('═' * 56)

            txt = self._comp_text
            txt.configure(state='normal')
            txt.delete('1.0', 'end')
            txt.insert('end', '\n'.join(lines))
            txt.configure(state='disabled')
        except Exception:
            pass

    def _plot_sensitivity(self):
        params_sens = [
            ('VL (V)',  self.var_VL,  self.var_VL.get()),
            ('R2 (Ω)', self.var_R2,  self.var_R2.get()),
            ('Xm (Ω)', self.var_Xm,  self.var_Xm.get()),
            ('HP',     self.var_HP,  self.var_HP.get()),
        ]
        sens_colors = [CLR_TITLE, CLR_RED, CLR_GRN, CLR_ORG]

        for ax, (param_lbl, var, base_val), col in zip(
                self.axes10.flat, params_sens, sens_colors):
            ax.cla()
            ax.set_facecolor(DARK_AX)
            perturbs = np.linspace(0.80 * base_val, 1.20 * base_val, 20)
            T_LR_list = []
            T_r_list  = []

            for pv in perturbs:
                var.set(pv)
                try:
                    _Ns, _Nr, _s, T_r2, _ = self._rated_quantities()
                    T_lr2 = self._torque_at_slip(1.0)
                    T_LR_list.append(T_lr2)
                    T_r_list.append(T_r2)
                except Exception:
                    T_LR_list.append(0.0)
                    T_r_list.append(0.0)
            var.set(base_val)   # restore

            p_pct = (perturbs / base_val - 1.0) * 100.0
            ax.plot(p_pct, T_LR_list, color=col,     lw=2,   label='T_LR')
            ax.plot(p_pct, T_r_list,  color=CLR_PUR, lw=2,   label='T_rated')
            ax.axvline(0, color='#6c7086', ls=':', lw=1)
            ax.set_title(f'{param_lbl} Sensitivity', color=CLR_TITLE, fontsize=9)
            ax.set_xlabel('Perturbation (%)', color=CLR_TEXT, fontsize=8)
            ax.set_ylabel('Torque (N·m)', color=CLR_TEXT, fontsize=8)
            ax.tick_params(colors=CLR_TEXT, labelsize=7)
            ax.grid(True, color=CLR_GRID, ls='--', alpha=0.7)
            ax.spines[:].set_color(CLR_SPINE)
            ax.legend(facecolor='#313244', edgecolor=CLR_SPINE,
                      labelcolor=CLR_TEXT, fontsize=7)

        self.fig10.tight_layout()
        self.canvas10.draw()

    def _export_summary(self):
        txt = self._comp_text
        txt.configure(state='normal')
        content = txt.get('1.0', 'end')
        txt.configure(state='disabled')
        print('\n' + '=' * 60)
        print('DESIGN D MOTOR SUITE — EXPORTED SUMMARY')
        print('=' * 60)
        print(content)
        print('=' * 60)

    # ══════════════════════════════════════════════════════════════════════════
    #  RESIZE HANDLER
    # ══════════════════════════════════════════════════════════════════════════
    def _on_resize(self, _event):
        for canvas in self._canvases:
            try:
                canvas.draw_idle()
            except Exception:
                pass


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    _app = DesignDMotorSuite(root)
    root.mainloop()


if __name__ == '__main__':
    main()
