"""
Triple Inverted Pendulum on a Cart — Physics Simulation
========================================================
Hardware context
----------------
  • Cart driven by a GoBilda lead-screw actuated by a 1120 RPM motor.
  • A second 312 RPM motor is available as an optional extra actuator.
  • Three rigid links are mounted above the cart in series.

Code structure
--------------
  Section 1 — SystemParameters   : all physical constants you may want to tweak.
  Section 2 — MotorModel         : GoBilda motor + lead-screw force model.
  Section 3 — TriplePendulumPhysics : Lagrangian equations of motion (sympy-derived,
                                      then lambdified for speed).
  Section 4 — PIDController      : simple PID with derivative filtering.
  Section 5 — Simulation         : numerical ODE integration via scipy.
  Section 6 — Visualizer         : Matplotlib animation + time-series plots.
  Section 7 — Test configurations: preset parameter sets for quick experiments.
  Section 8 — main()             : entry point; pick a test configuration and run.

Dependencies: numpy, scipy, matplotlib, sympy
  pip install numpy scipy matplotlib sympy
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SystemParameters
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemParameters:
    """
    All physical parameters for the triple-inverted-pendulum-on-cart system.

    Angles are measured from the upright vertical position:
      θ = 0   → perfectly balanced upright
      θ = π   → hanging downward

    How to modify:
      • Change link lengths (l1, l2, l3) in metres.
      • Change link masses (m1, m2, m3) in kilograms.
      • Change cart mass (M_cart) in kilograms.
      • Change cart travel limits (x_min, x_max) in metres.
      • Change gravity (g) if simulating other environments.
    """

    # ── Cart ─────────────────────────────────────────────────────────────────
    M_cart: float = 2.0          # Cart mass [kg]
    x_min: float = -0.5          # Left travel limit [m]
    x_max: float = 0.5           # Right travel limit [m]
    cart_friction: float = 5.0   # Linear viscous friction on the cart [N·s/m]

    # ── Link 1 (bottom link, attached to cart) ────────────────────────────────
    l1: float = 0.30             # Length [m]
    m1: float = 0.15             # Mass [kg]
    # Moment of inertia about pivot (uniform rod: m*l^2/3); override if needed
    I1: Optional[float] = None   # [kg·m²] — None → computed automatically

    # ── Link 2 (middle link) ──────────────────────────────────────────────────
    l2: float = 0.25             # Length [m]
    m2: float = 0.10             # Mass [kg]
    I2: Optional[float] = None

    # ── Link 3 (top link) ─────────────────────────────────────────────────────
    l3: float = 0.20             # Length [m]
    m3: float = 0.08             # Mass [kg]
    I3: Optional[float] = None

    # ── Link joint damping ────────────────────────────────────────────────────
    joint_damping: float = 0.001  # Viscous damping at each pivot [N·m·s/rad]

    # ── Environment ───────────────────────────────────────────────────────────
    g: float = 9.81              # Gravitational acceleration [m/s²]

    def __post_init__(self) -> None:
        """Auto-compute moments of inertia for uniform rods about their pivot."""
        if self.I1 is None:
            # Uniform rod rotated about one end: I = m*l²/3
            self.I1 = self.m1 * self.l1 ** 2 / 3.0
        if self.I2 is None:
            self.I2 = self.m2 * self.l2 ** 2 / 3.0
        if self.I3 is None:
            self.I3 = self.m3 * self.l3 ** 2 / 3.0

    def summary(self) -> str:
        lines = [
            "=== SystemParameters ===",
            f"  Cart: M={self.M_cart:.3f} kg, x∈[{self.x_min},{self.x_max}] m",
            f"  Link1: l={self.l1:.3f} m, m={self.m1:.3f} kg, I={self.I1:.5f} kg·m²",
            f"  Link2: l={self.l2:.3f} m, m={self.m2:.3f} kg, I={self.I2:.5f} kg·m²",
            f"  Link3: l={self.l3:.3f} m, m={self.m3:.3f} kg, I={self.I3:.5f} kg·m²",
            f"  g={self.g} m/s², cart_friction={self.cart_friction} N·s/m",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MotorModel
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MotorModel:
    """
    Models a GoBilda brushed DC motor driving a lead-screw cart actuator.

    Physics
    -------
    A DC motor has a linear torque-speed curve:
        τ(ω) = τ_stall * (1 - ω / ω_free)
    The lead-screw converts rotational motion to linear force:
        F = τ * 2π / lead_m * η
    where lead_m is the lead (m/rev) and η is mechanical efficiency.

    How to modify:
      • free_speed_rpm  — motor no-load RPM (1120 for primary, 312 for secondary)
      • stall_torque_Nm — motor stall torque (check GoBilda datasheet)
      • lead_mm         — lead-screw lead in millimetres (GoBilda default: 8 mm)
      • efficiency      — lead-screw mechanical efficiency (0–1)
    """

    # Primary drive motor (GoBilda Yellow Jacket, ~1120 RPM variant)
    free_speed_rpm: float = 1120.0    # Free (no-load) speed [RPM]
    stall_torque_Nm: float = 2.8      # Stall torque [N·m]

    # Lead-screw geometry
    lead_mm: float = 8.0             # Linear travel per revolution [mm]
    efficiency: float = 0.85         # Mechanical efficiency (friction losses)

    # Force / speed limits (physical safety bounds)
    max_force_N: float = 500.0       # Clamp applied force [N]
    max_speed_ms: float = None       # Computed in __post_init__

    def __post_init__(self) -> None:
        lead_m = self.lead_mm * 1e-3
        omega_free = self.free_speed_rpm * 2 * np.pi / 60.0   # rad/s
        # Max linear speed = free_speed * lead / (2π)
        self.max_speed_ms = omega_free * lead_m / (2 * np.pi)

    # ── Secondary motor (optional actuator, 312 RPM) ──────────────────────────
    @classmethod
    def secondary_motor(cls) -> "MotorModel":
        """
        Returns a MotorModel configured for the 312 RPM secondary actuator.
        The higher stall torque compensates for the lower speed.
        """
        return cls(
            free_speed_rpm=312.0,
            stall_torque_Nm=5.2,      # Higher-torque / lower-speed variant
            lead_mm=8.0,
            efficiency=0.85,
        )

    # ── Core calculation ──────────────────────────────────────────────────────
    def force_from_command(self, command: float, cart_speed_ms: float = 0.0) -> float:
        """
        Convert a normalised motor command (−1…+1) to a linear force [N].

        The motor command is analogous to a PWM duty-cycle fraction.
        Back-EMF from cart motion reduces the effective torque.

        Parameters
        ----------
        command      : Normalised input in [−1, +1].
        cart_speed_ms: Current cart speed [m/s] (for back-EMF calculation).

        Returns
        -------
        Force [N] applied to the cart (positive = rightward).
        """
        lead_m = self.lead_mm * 1e-3
        omega_free = self.free_speed_rpm * 2 * np.pi / 60.0   # rad/s

        # Requested rotational speed from command (linear → rotational)
        omega_demand = command * omega_free

        # Back-EMF from cart velocity
        omega_back = cart_speed_ms * 2 * np.pi / lead_m

        # Effective angular speed at motor shaft
        omega_eff = omega_demand - omega_back
        omega_eff = np.clip(omega_eff, -omega_free, omega_free)

        # Torque from linear motor model: τ = τ_stall * (1 - |ω_eff| / ω_free)
        sign = np.sign(omega_eff) if omega_eff != 0 else np.sign(command)
        tau = sign * self.stall_torque_Nm * (1.0 - abs(omega_eff) / omega_free)

        # Convert torque to linear force via lead-screw
        force = tau * 2 * np.pi / lead_m * self.efficiency

        # Clamp to physical limits
        return float(np.clip(force, -self.max_force_N, self.max_force_N))

    def summary(self) -> str:
        lines = [
            "=== MotorModel ===",
            f"  Free speed : {self.free_speed_rpm:.0f} RPM",
            f"  Stall torque: {self.stall_torque_Nm:.2f} N·m",
            f"  Lead        : {self.lead_mm:.1f} mm/rev",
            f"  Efficiency  : {self.efficiency:.0%}",
            f"  Max cart speed: {self.max_speed_ms:.3f} m/s",
            f"  Max force (stall): {self.force_from_command(1.0, 0.0):.1f} N",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TriplePendulumPhysics
# ══════════════════════════════════════════════════════════════════════════════

class TriplePendulumPhysics:
    """
    Derives and evaluates the Euler-Lagrange equations of motion for a
    cart + triple-inverted-pendulum system using SymPy, then compiles
    them to fast NumPy functions via lambdify.

    Generalised coordinates: q = [x, θ1, θ2, θ3]
      x  — cart position [m]
      θi — angle of link i from the upright vertical [rad]

    The equations of motion take the form:
        M(q) q̈ = τ(q, q̇, F_cart)
    where M is the mass matrix and τ is the generalised force vector.

    Derivation is done once at construction (~1-3 s); subsequent calls
    to equations_of_motion() are fast (microseconds).
    """

    def __init__(self, params: SystemParameters) -> None:
        self.params = params
        print("Deriving equations of motion (this takes a few seconds)…")
        t0 = time.time()
        self._build_eom()
        print(f"  Done in {time.time() - t0:.1f} s.")

    # ── Symbolic derivation ───────────────────────────────────────────────────
    def _build_eom(self) -> None:
        p = self.params

        # Symbolic time variable and generalised coordinates
        t = sp.Symbol("t")
        x, th1, th2, th3 = sp.symbols("x theta1 theta2 theta3", real=True)
        dx, dth1, dth2, dth3 = sp.symbols(
            "dx dtheta1 dtheta2 dtheta3", real=True
        )

        q = sp.Matrix([x, th1, th2, th3])
        dq = sp.Matrix([dx, dth1, dth2, dth3])

        g_sym = sp.Symbol("g", positive=True)
        g_val = p.g

        # ── Centre-of-mass positions (y measured upward from cart rail) ──────
        # Each link's CM is at the midpoint of its length.
        # Angles measured from upright (θ=0 ↔ straight up):
        #   x_cm = x_pivot + (l/2) * sin(θ)
        #   y_cm =           (l/2) * cos(θ)

        # Link 1 pivot is on the cart.
        x_cm1 = x + (p.l1 / 2) * sp.sin(th1)
        y_cm1 = (p.l1 / 2) * sp.cos(th1)

        # Link 2 pivot is at the tip of link 1.
        x_cm2 = x + p.l1 * sp.sin(th1) + (p.l2 / 2) * sp.sin(th2)
        y_cm2 = p.l1 * sp.cos(th1) + (p.l2 / 2) * sp.cos(th2)

        # Link 3 pivot is at the tip of link 2.
        x_cm3 = x + p.l1 * sp.sin(th1) + p.l2 * sp.sin(th2) + (p.l3 / 2) * sp.sin(th3)
        y_cm3 = p.l1 * sp.cos(th1) + p.l2 * sp.cos(th2) + (p.l3 / 2) * sp.cos(th3)

        # ── Velocities of each CM via Jacobian (∂pos/∂q · q̇) ──────────────
        pos_cm1 = sp.Matrix([x_cm1, y_cm1])
        pos_cm2 = sp.Matrix([x_cm2, y_cm2])
        pos_cm3 = sp.Matrix([x_cm3, y_cm3])

        def cm_vel(pos):
            jac = pos.jacobian(q)
            return jac * dq

        v_cm1 = cm_vel(pos_cm1)
        v_cm2 = cm_vel(pos_cm2)
        v_cm3 = cm_vel(pos_cm3)

        # ── Kinetic energy ───────────────────────────────────────────────────
        def v_sq(v):
            return (v.T * v)[0, 0]

        T_cart = sp.Rational(1, 2) * p.M_cart * dx ** 2
        T_link1 = sp.Rational(1, 2) * p.m1 * v_sq(v_cm1) + sp.Rational(1, 2) * p.I1 * dth1 ** 2
        T_link2 = sp.Rational(1, 2) * p.m2 * v_sq(v_cm2) + sp.Rational(1, 2) * p.I2 * dth2 ** 2
        T_link3 = sp.Rational(1, 2) * p.m3 * v_sq(v_cm3) + sp.Rational(1, 2) * p.I3 * dth3 ** 2
        T = sp.expand(T_cart + T_link1 + T_link2 + T_link3)

        # ── Potential energy (gravity, zero at cart rail) ────────────────────
        V = (p.m1 * g_val * y_cm1 + p.m2 * g_val * y_cm2 + p.m3 * g_val * y_cm3)
        V = sp.expand(V)

        # ── Mass matrix M: T = ½ q̇ᵀ M q̇  ────────────────────────────────
        # Extract quadratic form coefficients from T
        M_sym = sp.zeros(4, 4)
        for i in range(4):
            for j in range(4):
                # Coefficient of dq[i]*dq[j] in 2*T
                M_sym[i, j] = sp.diff(sp.diff(2 * T, dq[i]), dq[j])
        M_sym = sp.simplify(M_sym)

        # ── Christoffel / Coriolis-centrifugal vector C(q,q̇)·q̇ ──────────
        # Uses the formula:  C_i = Σ_j Σ_k Γ_{ijk} q̇_j q̇_k
        # Γ_{ijk} = ½ (∂M_{ij}/∂q_k + ∂M_{ik}/∂q_j − ∂M_{jk}/∂q_i)
        C_vec = sp.zeros(4, 1)
        for i in range(4):
            c_i = sp.Integer(0)
            for j in range(4):
                for k in range(4):
                    gamma = sp.Rational(1, 2) * (
                        sp.diff(M_sym[i, j], q[k])
                        + sp.diff(M_sym[i, k], q[j])
                        - sp.diff(M_sym[j, k], q[i])
                    )
                    c_i += gamma * dq[j] * dq[k]
            C_vec[i] = sp.expand(c_i)

        # ── Gravity vector G = ∂V/∂q ─────────────────────────────────────
        G_vec = sp.Matrix([sp.diff(V, qi) for qi in q])

        # ── Lambdify all symbolic expressions ────────────────────────────
        syms = [x, th1, th2, th3, dx, dth1, dth2, dth3]
        self._M_func = sp.lambdify(syms, M_sym, modules="numpy")
        self._C_func = sp.lambdify(syms, C_vec, modules="numpy")
        self._G_func = sp.lambdify(syms, G_vec, modules="numpy")

        # Store symbolic forms for inspection
        self._M_sym = M_sym
        self._G_sym = G_vec

    # ── Public interface ──────────────────────────────────────────────────────
    def equations_of_motion(
        self, state: np.ndarray, F_cart: float
    ) -> np.ndarray:
        """
        Compute state derivatives [q̇, q̈] for the ODE integrator.

        Parameters
        ----------
        state   : [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3]  (8-element array)
        F_cart  : External force applied to the cart [N] (positive = right)

        Returns
        -------
        dstate : [ẋ, θ̇1, θ̇2, θ̇3, ẍ, θ̈1, θ̈2, θ̈3]  (8-element array)
        """
        p = self.params
        x, th1, th2, th3, dx, dth1, dth2, dth3 = state
        args = (x, th1, th2, th3, dx, dth1, dth2, dth3)

        M = np.array(self._M_func(*args), dtype=float)
        C = np.array(self._C_func(*args), dtype=float).flatten()
        G = np.array(self._G_func(*args), dtype=float).flatten()

        # Damping forces: cart friction and joint damping
        D = np.array([
            p.cart_friction * dx,
            p.joint_damping * dth1,
            p.joint_damping * dth2,
            p.joint_damping * dth3,
        ])

        # Generalised force vector (force on cart maps to x coordinate only)
        Q = np.array([F_cart, 0.0, 0.0, 0.0])

        # Solve M q̈ = Q − C − G − D
        rhs = Q - C - G - D
        try:
            q_ddot = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            # Degenerate configuration — use least-squares fallback
            q_ddot, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

        # Enforce cart travel limits by zeroing acceleration at boundaries
        if (x <= p.x_min and dx < 0) or (x >= p.x_max and dx > 0):
            q_ddot[0] = 0.0

        return np.concatenate([state[4:], q_ddot])

    def tip_positions(self, state: np.ndarray) -> dict:
        """
        Return Cartesian (x, y) positions of each link's pivot and tip.

        Useful for animation and validation.
        """
        p = self.params
        x, th1, th2, th3 = state[:4]

        pivot0 = np.array([x, 0.0])

        tip1 = pivot0 + np.array([p.l1 * np.sin(th1), p.l1 * np.cos(th1)])
        tip2 = tip1 + np.array([p.l2 * np.sin(th2), p.l2 * np.cos(th2)])
        tip3 = tip2 + np.array([p.l3 * np.sin(th3), p.l3 * np.cos(th3)])

        return {
            "cart": pivot0,
            "tip1": tip1,
            "tip2": tip2,
            "tip3": tip3,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PIDController
# ══════════════════════════════════════════════════════════════════════════════

class PIDController:
    """
    Multi-variable PD/PID controller for cart-pendulum stabilisation.

    Strategy
    --------
    The total control force is a weighted sum of contributions from:
      1. Cart position error  (bring cart back to x=0)
      2. Each pendulum angle  (keep links upright)

    All gains default to zero — set only the ones you need.

    How to use:
      ctrl = PIDController(
          kp_x=50, kd_x=20,         # Cart position control
          kp_th=[200, 150, 80],      # Angle proportional gains
          kd_th=[40, 30, 15],        # Angle derivative gains
          ki_th=[1.0, 0.5, 0.2],     # Angle integral gains (small!)
          max_force=300,
      )
      force = ctrl.compute(state, dt, x_ref=0.0)
    """

    def __init__(
        self,
        kp_x: float = 0.0,
        ki_x: float = 0.0,
        kd_x: float = 0.0,
        kp_th: Optional[List[float]] = None,
        ki_th: Optional[List[float]] = None,
        kd_th: Optional[List[float]] = None,
        max_force: float = 300.0,
        derivative_filter_alpha: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        kp_x, ki_x, kd_x : PID gains for cart x-position.
        kp_th             : Proportional gains for [θ1, θ2, θ3].
        ki_th             : Integral gains   for [θ1, θ2, θ3].
        kd_th             : Derivative gains for [θ1, θ2, θ3].
        max_force         : Clamp on total output force [N].
        derivative_filter_alpha: Low-pass coefficient for derivative term (0–1).
                                  Smaller → more filtering of high-freq noise.
        """
        self.kp_x = kp_x
        self.ki_x = ki_x
        self.kd_x = kd_x
        self.kp_th = list(kp_th or [0.0, 0.0, 0.0])
        self.ki_th = list(ki_th or [0.0, 0.0, 0.0])
        self.kd_th = list(kd_th or [0.0, 0.0, 0.0])
        self.max_force = max_force
        self.alpha = derivative_filter_alpha

        # Internal state
        self._integral_x: float = 0.0
        self._integral_th: List[float] = [0.0, 0.0, 0.0]
        self._prev_error_x: float = 0.0
        self._prev_error_th: List[float] = [0.0, 0.0, 0.0]
        self._dfilt_x: float = 0.0
        self._dfilt_th: List[float] = [0.0, 0.0, 0.0]

    def reset(self) -> None:
        """Reset integrators and derivative memory."""
        self._integral_x = 0.0
        self._integral_th = [0.0, 0.0, 0.0]
        self._prev_error_x = 0.0
        self._prev_error_th = [0.0, 0.0, 0.0]
        self._dfilt_x = 0.0
        self._dfilt_th = [0.0, 0.0, 0.0]

    def compute(
        self, state: np.ndarray, dt: float, x_ref: float = 0.0
    ) -> float:
        """
        Compute the control force for the current time step.

        Parameters
        ----------
        state : [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3]
        dt    : Time step [s] since last call.
        x_ref : Desired cart position [m] (default 0.0).

        Returns
        -------
        force : Control force to apply to the cart [N].
        """
        x, th1, th2, th3, dx, dth1, dth2, dth3 = state
        angles = [th1, th2, th3]

        # ── Cart position PID ─────────────────────────────────────────────
        err_x = x_ref - x
        self._integral_x += err_x * dt
        raw_deriv_x = (err_x - self._prev_error_x) / (dt + 1e-12)
        self._dfilt_x = self.alpha * raw_deriv_x + (1 - self.alpha) * self._dfilt_x
        self._prev_error_x = err_x

        f_x = (
            self.kp_x * err_x
            + self.ki_x * self._integral_x
            + self.kd_x * self._dfilt_x
        )

        # ── Angle PID for each link (keep upright: θ=0) ───────────────────
        f_th = 0.0
        for i, (th, kp, ki, kd) in enumerate(
            zip(angles, self.kp_th, self.ki_th, self.kd_th)
        ):
            err_th = -th  # want θ → 0
            self._integral_th[i] += err_th * dt
            raw_d = (err_th - self._prev_error_th[i]) / (dt + 1e-12)
            self._dfilt_th[i] = (
                self.alpha * raw_d + (1 - self.alpha) * self._dfilt_th[i]
            )
            self._prev_error_th[i] = err_th

            f_th += (
                kp * err_th
                + ki * self._integral_th[i]
                + kd * self._dfilt_th[i]
            )

        total = f_x + f_th
        return float(np.clip(total, -self.max_force, self.max_force))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Simulation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    """Container for all time-series data produced by the simulation."""
    t:       np.ndarray   # Time [s]
    x:       np.ndarray   # Cart position [m]
    dx:      np.ndarray   # Cart velocity [m/s]
    theta1:  np.ndarray   # Link 1 angle [rad]
    theta2:  np.ndarray   # Link 2 angle [rad]
    theta3:  np.ndarray   # Link 3 angle [rad]
    dtheta1: np.ndarray   # Link 1 angular velocity [rad/s]
    dtheta2: np.ndarray   # Link 2 angular velocity [rad/s]
    dtheta3: np.ndarray   # Link 3 angular velocity [rad/s]
    force:   np.ndarray   # Applied control force [N]
    states:  np.ndarray   # Full 8-element state matrix (N × 8)


class Simulation:
    """
    Numerical integration of the triple-pendulum equations of motion.

    Uses scipy.integrate.solve_ivp with the RK45 method (adaptive step).
    The control law is evaluated at each time step.

    How to run a simulation:
      sim = Simulation(physics, controller, motor)
      result = sim.run(
          t_span=(0, 5),          # Simulate 5 seconds
          initial_state=...,      # 8-element array
          dt_output=0.01,         # Output every 10 ms
      )
    """

    def __init__(
        self,
        physics: TriplePendulumPhysics,
        controller: Optional[PIDController] = None,
        motor: Optional[MotorModel] = None,
    ) -> None:
        self.physics = physics
        self.controller = controller
        self.motor = motor

    def run(
        self,
        t_span: Tuple[float, float],
        initial_state: np.ndarray,
        dt_output: float = 0.01,
        use_motor_model: bool = True,
    ) -> SimulationResult:
        """
        Integrate the system forward in time.

        Parameters
        ----------
        t_span       : (t_start, t_end) in seconds.
        initial_state: [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3]
        dt_output    : Time resolution of stored output [s].
        use_motor_model : If True, pass controller command through MotorModel
                         (adds back-EMF / saturation effects).

        Returns
        -------
        SimulationResult with all time series.
        """
        p = self.physics.params
        t_out = np.arange(t_span[0], t_span[1], dt_output)
        forces = []
        prev_t = t_span[0]

        if self.controller:
            self.controller.reset()

        def ode_rhs(t: float, state: np.ndarray) -> np.ndarray:
            nonlocal prev_t
            dt = max(t - prev_t, 1e-9)
            prev_t = t

            # Compute control force
            if self.controller is not None:
                f_cmd = self.controller.compute(state, dt)
            else:
                f_cmd = 0.0

            if use_motor_model and self.motor is not None:
                # Convert force command to normalised motor command, then back
                max_f = self.motor.max_force_N
                command = np.clip(f_cmd / max_f, -1.0, 1.0)
                F_cart = self.motor.force_from_command(command, cart_speed_ms=state[4])
            else:
                F_cart = f_cmd

            forces.append((t, F_cart))
            return self.physics.equations_of_motion(state, F_cart)

        print(f"Integrating t=[{t_span[0]}, {t_span[1]}] s …")
        sol = solve_ivp(
            ode_rhs,
            t_span,
            initial_state,
            method="RK45",
            t_eval=t_out,
            rtol=1e-5,
            atol=1e-7,
            max_step=dt_output,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        # Match forces to output time points (nearest)
        force_times = np.array([f[0] for f in forces])
        force_vals  = np.array([f[1] for f in forces])
        idx = np.searchsorted(force_times, sol.t).clip(0, len(force_vals) - 1)
        matched_forces = force_vals[idx]

        states = sol.y.T  # shape (N, 8)
        return SimulationResult(
            t=sol.t,
            x=states[:, 0],
            theta1=states[:, 1],
            theta2=states[:, 2],
            theta3=states[:, 3],
            dx=states[:, 4],
            dtheta1=states[:, 5],
            dtheta2=states[:, 6],
            dtheta3=states[:, 7],
            force=matched_forces,
            states=states,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Visualizer
# ══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """
    Matplotlib-based visualisation of the triple-pendulum simulation.

    Provides:
      • animate()          — real-time (or saved) animation of pendulum motion.
      • plot_time_series() — multi-panel time-series plots of angles, cart, force.
    """

    def __init__(self, physics: TriplePendulumPhysics, result: SimulationResult) -> None:
        self.physics = physics
        self.result = result
        self._p = physics.params

    # ── Animation ─────────────────────────────────────────────────────────────
    def animate(
        self,
        interval_ms: int = 20,
        save_path: Optional[str] = None,
        speed_factor: float = 1.0,
    ) -> animation.FuncAnimation:
        """
        Create an animation of the pendulum swinging on the cart.

        Parameters
        ----------
        interval_ms  : Delay between frames in real time [ms].
        save_path    : If given, save animation to this file (e.g. 'anim.gif').
        speed_factor : >1 → faster playback; <1 → slow-motion.

        Returns
        -------
        anim : FuncAnimation object (display with plt.show() in a notebook).
        """
        r = self.result
        p = self._p
        total_len = p.l1 + p.l2 + p.l3
        margin = 0.15

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlim(p.x_min - margin, p.x_max + margin)
        ax.set_ylim(-0.2, total_len + margin)
        ax.set_aspect("equal")
        ax.set_xlabel("Cart position x [m]")
        ax.set_ylabel("Height [m]")
        ax.set_title("Triple Inverted Pendulum — Animation")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(p.x_min, color="salmon", linewidth=1, linestyle=":")
        ax.axvline(p.x_max, color="salmon", linewidth=1, linestyle=":")

        # Track / rail
        ax.plot([p.x_min, p.x_max], [0, 0], "k-", linewidth=4, zorder=1)

        # Cart rectangle
        cart_w, cart_h = 0.12, 0.06
        cart_patch = plt.Rectangle(
            (0 - cart_w / 2, -cart_h), cart_w, cart_h,
            fc="steelblue", ec="navy", zorder=3
        )
        ax.add_patch(cart_patch)

        # Pendulum links
        colors = ["tomato", "goldenrod", "mediumseagreen"]
        link_lines = [
            ax.plot([], [], "-o", color=c, linewidth=3, markersize=6, zorder=4)[0]
            for c in colors
        ]
        pivot_dot = ax.plot([], [], "ko", markersize=8, zorder=5)[0]

        # Time label
        time_text = ax.text(
            0.02, 0.96, "", transform=ax.transAxes, fontsize=10, va="top"
        )

        # Angle readouts
        angle_text = ax.text(
            0.70, 0.96, "", transform=ax.transAxes, fontsize=9, va="top",
            family="monospace"
        )

        # Subsample for animation frame rate
        n_frames = len(r.t)
        step = max(1, int(n_frames / (r.t[-1] * 1000 / interval_ms / speed_factor)))

        def init():
            pivot_dot.set_data([], [])
            for ln in link_lines:
                ln.set_data([], [])
            time_text.set_text("")
            angle_text.set_text("")
            return link_lines + [cart_patch, pivot_dot, time_text, angle_text]

        def update(frame_idx: int):
            i = frame_idx * step
            if i >= n_frames:
                i = n_frames - 1
            state = r.states[i]
            tips = self.physics.tip_positions(state)

            cart_x = tips["cart"][0]
            cart_patch.set_xy((cart_x - cart_w / 2, -cart_h))
            pivot_dot.set_data([cart_x], [0.0])

            # Link 1
            link_lines[0].set_data(
                [cart_x, tips["tip1"][0]], [0.0, tips["tip1"][1]]
            )
            # Link 2
            link_lines[1].set_data(
                [tips["tip1"][0], tips["tip2"][0]],
                [tips["tip1"][1], tips["tip2"][1]],
            )
            # Link 3
            link_lines[2].set_data(
                [tips["tip2"][0], tips["tip3"][0]],
                [tips["tip2"][1], tips["tip3"][1]],
            )

            time_text.set_text(f"t = {r.t[i]:.3f} s")
            angle_text.set_text(
                f"θ1={np.degrees(state[1]):+6.1f}°\n"
                f"θ2={np.degrees(state[2]):+6.1f}°\n"
                f"θ3={np.degrees(state[3]):+6.1f}°"
            )
            return link_lines + [cart_patch, pivot_dot, time_text, angle_text]

        n_anim_frames = n_frames // step
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=n_anim_frames,
            init_func=init,
            interval=interval_ms,
            blit=True,
        )

        if save_path:
            print(f"Saving animation to {save_path} …")
            writer = (
                animation.PillowWriter(fps=int(1000 / interval_ms))
                if save_path.endswith(".gif")
                else animation.FFMpegWriter(fps=int(1000 / interval_ms))
            )
            anim.save(save_path, writer=writer)
            print("  Saved.")

        return anim

    # ── Time-series plots ─────────────────────────────────────────────────────
    def plot_time_series(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot angles, cart position/velocity, and control force vs. time.

        Parameters
        ----------
        save_path : If given, save the figure to this path.
        """
        r = self.result
        t = r.t

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("Triple Inverted Pendulum — Time Series", fontsize=13)

        # ── Panel 1: angles ───────────────────────────────────────────────
        ax = axes[0]
        ax.plot(t, np.degrees(r.theta1), label="θ1 (bottom)", color="tomato")
        ax.plot(t, np.degrees(r.theta2), label="θ2 (middle)", color="goldenrod")
        ax.plot(t, np.degrees(r.theta3), label="θ3 (top)",    color="mediumseagreen")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Angle [°]")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Panel 2: angular velocities ───────────────────────────────────
        ax = axes[1]
        ax.plot(t, np.degrees(r.dtheta1), label="θ̇1", color="tomato",      linestyle="--")
        ax.plot(t, np.degrees(r.dtheta2), label="θ̇2", color="goldenrod",   linestyle="--")
        ax.plot(t, np.degrees(r.dtheta3), label="θ̇3", color="mediumseagreen", linestyle="--")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Angular vel. [°/s]")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Panel 3: cart position and velocity ───────────────────────────
        ax = axes[2]
        ax.plot(t, r.x,  label="x (position)", color="steelblue")
        ax.plot(t, r.dx, label="ẋ (velocity)",  color="steelblue", linestyle="--", alpha=0.7)
        ax.axhline(0,          color="gray",   linewidth=0.8, linestyle="--")
        ax.axhline(self._p.x_min, color="salmon", linewidth=0.8, linestyle=":")
        ax.axhline(self._p.x_max, color="salmon", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Cart x [m] / ẋ [m/s]")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Panel 4: control force ────────────────────────────────────────
        ax = axes[3]
        ax.plot(t, r.force, label="F_cart", color="purple")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Force [N]")
        ax.set_xlabel("Time [s]")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=120)
            print(f"Time-series plot saved to {save_path}")

        return fig


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Test Configurations
# ══════════════════════════════════════════════════════════════════════════════

def config_free_swing() -> Tuple[SystemParameters, MotorModel, PIDController, np.ndarray]:
    """
    Configuration A — Free swing (no control).

    All three links start from a nearly-upright position with a small
    perturbation. No control force is applied so the pendulum falls freely.
    Use this to verify that the physics are correct (energy conservation, etc.).
    """
    params = SystemParameters(
        M_cart=2.0,
        l1=0.30, m1=0.15,
        l2=0.25, m2=0.10,
        l3=0.20, m3=0.08,
        cart_friction=0.5,    # Low friction for free swing
        joint_damping=0.0001,
    )
    motor = MotorModel()
    controller = None  # type: ignore[assignment]

    # Initial state: small perturbation from upright
    initial_state = np.array([
        0.0,           # x
        np.radians(3), # θ1: 3° from vertical
        np.radians(2), # θ2
        np.radians(1), # θ3
        0.0, 0.0, 0.0, 0.0,  # all velocities zero
    ])
    return params, motor, controller, initial_state


def config_pd_stabilise() -> Tuple[SystemParameters, MotorModel, PIDController, np.ndarray]:
    """
    Configuration B — PD stabilisation near the upright.

    A PD controller tries to keep all links upright and the cart at x=0.
    The gains are tuned for the default link dimensions; increase if the
    pendulum falls too quickly, decrease if it oscillates violently.

    Changing gains:
      kp_th  — higher values pull links back to vertical faster.
      kd_th  — higher values damp oscillations around upright.
      kp_x   — cart position stiffness.
      kd_x   — cart velocity damping.
    """
    params = SystemParameters(
        M_cart=2.0,
        l1=0.30, m1=0.15,
        l2=0.25, m2=0.10,
        l3=0.20, m3=0.08,
    )
    motor = MotorModel(free_speed_rpm=1120.0, stall_torque_Nm=2.8)
    controller = PIDController(
        kp_x=60.0,   kd_x=25.0,
        kp_th=[350.0, 250.0, 120.0],
        kd_th=[55.0,  40.0,  20.0],
        ki_th=[0.5,   0.3,   0.1],
        max_force=300.0,
    )
    # Initial state: small pushes on each link
    initial_state = np.array([
        0.0,
        np.radians(5),
        np.radians(4),
        np.radians(3),
        0.0, 0.0, 0.0, 0.0,
    ])
    return params, motor, controller, initial_state


def config_longer_links() -> Tuple[SystemParameters, MotorModel, PIDController, np.ndarray]:
    """
    Configuration C — Longer, heavier links.

    Tests how the simulation handles a different link geometry.
    Longer links are harder to stabilise; gains may need tuning.
    """
    params = SystemParameters(
        M_cart=3.0,
        l1=0.50, m1=0.25,
        l2=0.45, m2=0.20,
        l3=0.40, m3=0.15,
        x_min=-0.8, x_max=0.8,
        cart_friction=6.0,
    )
    motor = MotorModel(free_speed_rpm=1120.0, stall_torque_Nm=2.8)
    controller = PIDController(
        kp_x=40.0,  kd_x=15.0,
        kp_th=[500.0, 350.0, 180.0],
        kd_th=[80.0,  60.0,  30.0],
        max_force=400.0,
    )
    initial_state = np.array([
        0.0,
        np.radians(4),
        np.radians(3),
        np.radians(2),
        0.0, 0.0, 0.0, 0.0,
    ])
    return params, motor, controller, initial_state


def config_secondary_motor() -> Tuple[SystemParameters, MotorModel, PIDController, np.ndarray]:
    """
    Configuration D — Use the 312 RPM secondary motor.

    The secondary motor has more torque but lower speed. It is better suited
    when large forces are needed for heavy payloads at slower cart speeds.
    """
    params = SystemParameters(
        M_cart=4.0,
        l1=0.30, m1=0.30,
        l2=0.25, m2=0.25,
        l3=0.20, m3=0.20,
        cart_friction=8.0,
    )
    motor = MotorModel.secondary_motor()
    controller = PIDController(
        kp_x=70.0,  kd_x=30.0,
        kp_th=[400.0, 280.0, 140.0],
        kd_th=[65.0,  48.0,  24.0],
        max_force=400.0,
    )
    initial_state = np.array([
        0.0,
        np.radians(6),
        np.radians(4),
        np.radians(2),
        0.0, 0.0, 0.0, 0.0,
    ])
    return params, motor, controller, initial_state


# Map of named configurations for easy selection on the command line
CONFIGURATIONS = {
    "free_swing":       config_free_swing,
    "pd_stabilise":     config_pd_stabilise,
    "longer_links":     config_longer_links,
    "secondary_motor":  config_secondary_motor,
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — main()
# ══════════════════════════════════════════════════════════════════════════════

def main(
    config_name: str = "pd_stabilise",
    t_end: float = 5.0,
    dt_output: float = 0.01,
    animate: bool = True,
    save_animation: Optional[str] = None,
    save_plot: Optional[str] = None,
    show_plots: bool = True,
) -> SimulationResult:
    """
    Run the triple-inverted-pendulum simulation with a named configuration.

    Parameters
    ----------
    config_name    : One of the keys in CONFIGURATIONS.
    t_end          : Simulation end time [s].
    dt_output      : Output resolution [s].
    animate        : Show Matplotlib animation after simulation.
    save_animation : File path to save animation (e.g. 'anim.gif'), or None.
    save_plot      : File path to save time-series figure, or None.
    show_plots     : If True, call plt.show() at the end.

    Returns
    -------
    SimulationResult : All time-series data.

    How to add a new configuration:
      1. Write a function config_mytest() → (SystemParameters, MotorModel,
         PIDController, initial_state).
      2. Add it to the CONFIGURATIONS dict.
      3. Call main(config_name='mytest').
    """
    if config_name not in CONFIGURATIONS:
        raise ValueError(
            f"Unknown configuration '{config_name}'. "
            f"Choose one of: {list(CONFIGURATIONS.keys())}"
        )

    print(f"\n{'='*60}")
    print(f"  Triple Inverted Pendulum Simulation")
    print(f"  Configuration: {config_name}")
    print(f"{'='*60}\n")

    # ── Load configuration ────────────────────────────────────────────────────
    params, motor, controller, initial_state = CONFIGURATIONS[config_name]()

    print(params.summary())
    print()
    print(motor.summary())
    print()
    if controller is not None:
        print("PID Controller enabled.")
    else:
        print("No controller — free swing.")
    print()

    # ── Build physics (derives EOM symbolically) ──────────────────────────────
    physics = TriplePendulumPhysics(params)

    # ── Run simulation ────────────────────────────────────────────────────────
    sim = Simulation(physics, controller, motor)
    result = sim.run(
        t_span=(0.0, t_end),
        initial_state=initial_state,
        dt_output=dt_output,
        use_motor_model=(controller is not None),
    )
    print(f"  Integration complete. {len(result.t)} output points.\n")

    # ── Print summary statistics ──────────────────────────────────────────────
    print("Results summary:")
    print(f"  Cart x range: [{result.x.min():.4f}, {result.x.max():.4f}] m")
    print(f"  θ1 range: [{np.degrees(result.theta1.min()):.1f}°, "
          f"{np.degrees(result.theta1.max()):.1f}°]")
    print(f"  θ2 range: [{np.degrees(result.theta2.min()):.1f}°, "
          f"{np.degrees(result.theta2.max()):.1f}°]")
    print(f"  θ3 range: [{np.degrees(result.theta3.min()):.1f}°, "
          f"{np.degrees(result.theta3.max()):.1f}°]")
    print(f"  Max |F_cart|: {np.max(np.abs(result.force)):.1f} N")

    # ── Visualise ─────────────────────────────────────────────────────────────
    viz = Visualizer(physics, result)

    # Time-series plot (always generated)
    fig_ts = viz.plot_time_series(save_path=save_plot)

    # Animation
    anim_obj = None
    if animate:
        anim_obj = viz.animate(
            interval_ms=20,
            save_path=save_animation,
            speed_factor=1.0,
        )

    if show_plots:
        plt.show()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Script entry-point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Triple Inverted Pendulum Physics Simulation"
    )
    parser.add_argument(
        "--config",
        default="pd_stabilise",
        choices=list(CONFIGURATIONS.keys()),
        help="Named test configuration to run (default: pd_stabilise).",
    )
    parser.add_argument(
        "--t_end", type=float, default=5.0,
        help="Simulation duration in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--dt", type=float, default=0.01,
        help="Output time resolution in seconds (default: 0.01).",
    )
    parser.add_argument(
        "--no-animate", action="store_true",
        help="Skip the animation (only show time-series plot).",
    )
    parser.add_argument(
        "--save-anim", type=str, default=None,
        help="Save animation to this file (e.g. anim.gif).",
    )
    parser.add_argument(
        "--save-plot", type=str, default=None,
        help="Save time-series figure to this file (e.g. plot.png).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not call plt.show() (useful for headless / script use).",
    )
    args = parser.parse_args()

    main(
        config_name=args.config,
        t_end=args.t_end,
        dt_output=args.dt,
        animate=not args.no_animate,
        save_animation=args.save_anim,
        save_plot=args.save_plot,
        show_plots=not args.no_show,
    )
