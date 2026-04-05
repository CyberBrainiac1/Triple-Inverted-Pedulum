"""
Triple Inverted Pendulum on a Cart — Physics Simulation
========================================================
Hardware context
----------------
  • Cart driven by a brushed DC lead-screw actuator (1120 RPM free-run, 2.8 N·m stall torque, 8 mm/rev lead).
  • A second 312 RPM motor is available as an optional extra actuator.
  • Three rigid links are mounted above the cart in series.

Code structure
--------------
  Section 1 — SystemParameters      : all physical constants you may want to tweak.
  Section 2 — MotorModel            : brushed DC motor + lead-screw force model.
  Section 3 — TriplePendulumPhysics : Lagrangian equations of motion, derived by hand
                                      and implemented as fast NumPy arithmetic.
  Section 4 — PIDController         : simple PID with derivative filtering.
  Section 5 — Simulation            : numerical ODE integration via scipy.
  Section 6 — Visualizer            : Matplotlib animation + time-series plots.
  Section 7 — Test configurations   : preset parameter sets for quick experiments.
  Section 8 — main()                : entry point; pick a test configuration and run.

Physics overview (see README.md for full explanation)
-----------------------------------------------------
  The system has 4 degrees of freedom: cart position x, and three link angles
  θ1, θ2, θ3 measured from the upright vertical.

  The Euler-Lagrange equations give us one equation per degree of freedom:

      M(q) · q̈  =  Q  −  h(q, q̇)  −  G(q)  −  D(q̇)

  where:
    M   = 4×4 mass matrix (encodes how inertia couples all four bodies together)
    q̈  = [ẍ, θ̈1, θ̈2, θ̈3]  (the accelerations we are solving for)
    Q   = [F_cart, 0, 0, 0]   (external force on the cart only)
    h   = Coriolis/centrifugal force vector (speed-squared coupling terms)
    G   = gravity force vector  (∂V/∂q — destabilises an inverted pendulum)
    D   = viscous damping vector (cart friction + pivot damping)

  All entries of M, h, G are derived analytically in Section 3.
  No symbolic algebra library is needed — just standard NumPy.

Dependencies: numpy, scipy, matplotlib
  pip install numpy scipy matplotlib
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
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
    Models a brushed DC motor driving a lead-screw cart actuator.

    Physics
    -------
    A DC motor has a linear torque-speed curve:
        τ(ω) = τ_stall * (1 - ω / ω_free)
    The lead-screw converts rotational motion to linear force:
        F = τ * 2π / lead_m * η
    where lead_m is the lead (m/rev) and η is mechanical efficiency.

    How to modify:
      • free_speed_rpm  — motor no-load RPM (1120 for primary, 312 for secondary)
      • stall_torque_Nm — motor stall torque (N·m)
      • lead_mm         — lead-screw lead in millimetres (default: 8 mm)
      • efficiency      — lead-screw mechanical efficiency (0–1)
    """

    # Primary drive motor (~1120 RPM free-run, 2.8 N·m stall torque)
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
        Convert a normalised motor voltage command (−1…+1) to a linear force [N].

        DC motor physics
        ----------------
        A brushed DC motor follows the linear torque-speed curve:
            τ = τ_stall · (V/V_max  −  ω_shaft / ω_free)

        where V/V_max = command (normalised applied voltage = PWM duty cycle).

        At stall  (ω_shaft = 0):  τ = τ_stall · command  → maximum torque.
        At free run (no load):    τ = 0, ω_shaft = command · ω_free.
        Back-EMF from the cart moving reduces the effective torque.

        The lead-screw converts shaft torque to linear cart force:
            F = τ · (2π / lead_m) · η
        where lead_m = lead in metres, η = mechanical efficiency.

        Parameters
        ----------
        command      : Normalised voltage in [−1, +1].  +1 = full forward.
        cart_speed_ms: Current cart speed [m/s] (used to compute back-EMF).

        Returns
        -------
        Force [N] applied to the cart (positive = rightward).
        """
        lead_m     = self.lead_mm * 1e-3
        omega_free = self.free_speed_rpm * 2.0 * np.pi / 60.0   # rad/s

        # Current shaft angular speed, inferred from the cart's linear speed
        # (the lead-screw is the mechanical link between them)
        omega_shaft = cart_speed_ms * 2.0 * np.pi / lead_m

        # Torque from the DC motor torque-speed characteristic:
        #   τ = τ_stall * (command - ω_shaft / ω_free)
        # Clamp to ±τ_stall so we cannot exceed physical stall torque.
        tau = self.stall_torque_Nm * (command - omega_shaft / omega_free)
        tau = float(np.clip(tau, -self.stall_torque_Nm, self.stall_torque_Nm))

        # Convert rotational torque [N·m] to linear force [N] via lead-screw
        force = tau * 2.0 * np.pi / lead_m * self.efficiency

        # Clamp to the user-set safety limit
        return float(np.clip(force, -self.max_force_N, self.max_force_N))

    def stall_force(self) -> float:
        """Maximum linear force [N] this motor/lead-screw can produce (at stall)."""
        lead_m = self.lead_mm * 1e-3
        return self.stall_torque_Nm * 2.0 * np.pi / lead_m * self.efficiency

    def summary(self) -> str:
        lines = [
            "=== MotorModel ===",
            f"  Free speed   : {self.free_speed_rpm:.0f} RPM",
            f"  Stall torque : {self.stall_torque_Nm:.2f} N·m",
            f"  Lead         : {self.lead_mm:.1f} mm/rev",
            f"  Efficiency   : {self.efficiency:.0%}",
            f"  Max cart speed (free-run): {self.max_speed_ms:.3f} m/s",
            f"  Max force    (at stall)  : {self.stall_force():.1f} N",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TriplePendulumPhysics
# ══════════════════════════════════════════════════════════════════════════════

class TriplePendulumPhysics:
    """
    Evaluates the Euler-Lagrange equations of motion for a cart +
    triple-inverted-pendulum system using only NumPy arithmetic.

    --- PHYSICS DERIVATION SUMMARY ---

    Generalised coordinates:  q = [x, θ1, θ2, θ3]
      x  — cart horizontal position [m]
      θi — angle of link i measured FROM THE UPRIGHT VERTICAL [rad]
           (θ = 0 → perfectly balanced upright; θ = π → hanging down)

    The system has 4 bodies: the cart plus three rigid links.
    Each link i has:
      length   li, mass mi, moment of inertia Ii = mi*li²/3 (uniform rod)
      half-length  ai = li / 2  (centre-of-mass is at the midpoint)

    ── Step 1: Positions of each centre of mass ─────────────────────────────
    Link pivots stack from the cart upward.  Measuring from the cart pivot:

      CM1 = ( x  +  a1·sin θ1,           a1·cos θ1 )
      CM2 = ( x  + l1·sin θ1 + a2·sin θ2,  l1·cos θ1 + a2·cos θ2 )
      CM3 = ( x  + l1·sin θ1 + l2·sin θ2 + a3·sin θ3,
                   l1·cos θ1 + l2·cos θ2 + a3·cos θ3 )

    ── Step 2: Kinetic energy T ─────────────────────────────────────────────
    T = ½ M_cart·ẋ²
      + ½ m1·(ẋCM1² + ẏCM1²)  + ½ I1·θ̇1²
      + ½ m2·(ẋCM2² + ẏCM2²)  + ½ I2·θ̇2²
      + ½ m3·(ẋCM3² + ẏCM3²)  + ½ I3·θ̇3²

    Expanding the squared velocity terms and grouping by pairs of
    generalised velocities (ẋ, θ̇i) gives the mass matrix M such that:
        T = ½ q̇ᵀ M q̇

    ── Step 3: Potential energy V ───────────────────────────────────────────
    Only gravity matters (y measured upward from the cart rail):
        V = m1·g·a1·cos θ1
          + m2·g·(l1·cos θ1 + a2·cos θ2)
          + m3·g·(l1·cos θ1 + l2·cos θ2 + a3·cos θ3)

    ── Step 4: Euler-Lagrange equations ─────────────────────────────────────
    For each coordinate qi:   d/dt(∂L/∂q̇i) − ∂L/∂qi = Qi_external
    with Lagrangian L = T − V.  Rearranging yields:

        M(q) · q̈  =  Q  −  h(q, q̇)  −  G(q)  −  D(q̇)

    where:
      Q = [F_cart, 0, 0, 0]  — external force (only on the cart)
      h = Coriolis/centrifugal vector  (θ̇² coupling terms, derived below)
      G = gravity vector = ∂V/∂q      (destabilises the inverted pendulum)
      D = damping vector               (viscous friction on cart and pivots)

    ── Step 5: Analytical mass matrix (derived by expanding T) ──────────────

    Define shorthand coupling coefficients:
      α1  = m1·a1 + (m2+m3)·l1    — total effective mass pulling on θ1 pivot
      α2  = m2·a2 +  m3·l2        — total effective mass pulling on θ2 pivot
      α3  = m3·a3                  — mass at θ3 pivot
      β12 = m2·l1·a2 + m3·l1·l2   — coupling inertia between links 1 and 2
      β13 = m3·l1·a3               — coupling inertia between links 1 and 3
      β23 = m3·l2·a3               — coupling inertia between links 2 and 3

    The 4×4 mass matrix (symmetric, so only upper-triangle shown):

        M[0,0] = M_cart + m1 + m2 + m3          (total system mass → ẍ)
        M[0,1] = α1·cos θ1                       (cart–link1 coupling)
        M[0,2] = α2·cos θ2                       (cart–link2 coupling)
        M[0,3] = α3·cos θ3                       (cart–link3 coupling)
        M[1,1] = m1·a1² + I1 + (m2+m3)·l1²     (link1 self-inertia)
        M[1,2] = β12·cos(θ1−θ2)                 (link1–link2 coupling)
        M[1,3] = β13·cos(θ1−θ3)                 (link1–link3 coupling)
        M[2,2] = m2·a2² + I2 + m3·l2²           (link2 self-inertia)
        M[2,3] = β23·cos(θ2−θ3)                 (link2–link3 coupling)
        M[3,3] = m3·a3² + I3                     (link3 self-inertia)

    ── Step 6: Coriolis/centrifugal vector h ────────────────────────────────
    These terms arise from differentiating the angle-dependent entries of
    M with respect to time.  Using the Christoffel-symbol formula
        Γ_{ijk} = ½(∂M_{ij}/∂q_k + ∂M_{ik}/∂q_j − ∂M_{jk}/∂q_i)
        h[i] = Σ_{j,k} Γ_{ijk}·q̇_j·q̇_k
    and applying it to our M gives:

        h[0] = −α1·sin θ1·θ̇1²  − α2·sin θ2·θ̇2²  − α3·sin θ3·θ̇3²
        h[1] =  β12·sin(θ1−θ2)·θ̇2²  +  β13·sin(θ1−θ3)·θ̇3²
        h[2] = −β12·sin(θ1−θ2)·θ̇1²  +  β23·sin(θ2−θ3)·θ̇3²
        h[3] = −β13·sin(θ1−θ3)·θ̇1²  −  β23·sin(θ2−θ3)·θ̇2²

    Physical meaning of h[0]: when the pendulum rotates (θ̇i ≠ 0) the
    pivot swings outward, creating a centripetal force that is felt as a
    horizontal force on the cart — this pushes the cart sideways.

    ── Step 7: Gravity vector G = ∂V/∂q ────────────────────────────────────
        G[0] = 0                     (V does not depend on cart position x)
        G[1] = −α1·g·sin θ1         (gravity tips link 1 further over)
        G[2] = −α2·g·sin θ2
        G[3] = −α3·g·sin θ3

    Note the negative signs: at θ = 0 (upright) G = 0 (no torque).
    For θ > 0, G < 0 so the equation M·q̈ = Q − h − G gives a positive
    (destabilising) angular acceleration — gravity makes things worse.
    This is why a controller is needed!
    """

    def __init__(self, params: SystemParameters) -> None:
        self.params = params
        p = params

        # ── Pre-compute constant coupling coefficients ────────────────────────
        # These depend only on masses and lengths, not on the angle state.
        a1, a2, a3 = p.l1 / 2.0, p.l2 / 2.0, p.l3 / 2.0

        # α_i  — coupling between the cart translation and rotation of link i
        self._a1 = a1
        self._a2 = a2
        self._a3 = a3
        self._alpha1 = p.m1 * a1 + (p.m2 + p.m3) * p.l1
        self._alpha2 = p.m2 * a2 +  p.m3         * p.l2
        self._alpha3 = p.m3 * a3

        # β_ij — coupling between rotation of link i and link j
        self._beta12 = p.m2 * p.l1 * a2 + p.m3 * p.l1 * p.l2
        self._beta13 = p.m3 * p.l1 * a3
        self._beta23 = p.m3 * p.l2 * a3

        # Constant diagonal entries of M (do not depend on angles)
        self._M00 = p.M_cart + p.m1 + p.m2 + p.m3
        self._M11 = p.m1 * a1 ** 2 + p.I1 + (p.m2 + p.m3) * p.l1 ** 2
        self._M22 = p.m2 * a2 ** 2 + p.I2 +  p.m3          * p.l2 ** 2
        self._M33 = p.m3 * a3 ** 2 + p.I3

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _mass_matrix(self, th1: float, th2: float, th3: float) -> np.ndarray:
        """
        Build the 4×4 symmetric mass matrix M(θ1, θ2, θ3).

        Each off-diagonal entry couples two coordinates.  The cosine factors
        appear because the projection of one link's acceleration onto another
        depends on the angle between them.
        """
        c1  = np.cos(th1)
        c2  = np.cos(th2)
        c3  = np.cos(th3)
        c12 = np.cos(th1 - th2)   # angle between link 1 and link 2
        c13 = np.cos(th1 - th3)   # angle between link 1 and link 3
        c23 = np.cos(th2 - th3)   # angle between link 2 and link 3

        M = np.array([
            # row 0 — cart (x)
            [self._M00,
             self._alpha1 * c1,
             self._alpha2 * c2,
             self._alpha3 * c3],
            # row 1 — link 1 (θ1)
            [self._alpha1 * c1,
             self._M11,
             self._beta12 * c12,
             self._beta13 * c13],
            # row 2 — link 2 (θ2)
            [self._alpha2 * c2,
             self._beta12 * c12,
             self._M22,
             self._beta23 * c23],
            # row 3 — link 3 (θ3)
            [self._alpha3 * c3,
             self._beta13 * c13,
             self._beta23 * c23,
             self._M33],
        ])
        return M

    def _coriolis_vector(
        self,
        th1: float, th2: float, th3: float,
        dth1: float, dth2: float, dth3: float,
    ) -> np.ndarray:
        """
        Compute the Coriolis/centrifugal vector h(q, q̇).

        These are the "speed-squared" terms that arise when rotating bodies
        couple to each other.  They are zero when all angular velocities are
        zero (i.e. at rest), and grow quadratically with speed.

        Derivation: differentiate the angle-dependent entries of M with
        respect to time, collect terms in q̇_j·q̇_k, apply Christoffel formula.
        """
        s1  = np.sin(th1)
        s2  = np.sin(th2)
        s3  = np.sin(th3)
        s12 = np.sin(th1 - th2)   # sin of angle difference between links 1 & 2
        s13 = np.sin(th1 - th3)
        s23 = np.sin(th2 - th3)

        # h[0]: how spinning links push the cart sideways (centripetal reaction)
        h0 = ( -self._alpha1 * s1  * dth1 ** 2
               -self._alpha2 * s2  * dth2 ** 2
               -self._alpha3 * s3  * dth3 ** 2 )

        # h[1]: how links 2 & 3 spinning affect link 1's pivot torque
        h1 = (  self._beta12 * s12 * dth2 ** 2
              + self._beta13 * s13 * dth3 ** 2 )

        # h[2]: how links 1 & 3 spinning affect link 2's pivot torque
        h2 = ( -self._beta12 * s12 * dth1 ** 2
              + self._beta23 * s23 * dth3 ** 2 )

        # h[3]: how links 1 & 2 spinning affect link 3's pivot torque
        h3 = ( -self._beta13 * s13 * dth1 ** 2
               -self._beta23 * s23 * dth2 ** 2 )

        return np.array([h0, h1, h2, h3])

    def _gravity_vector(
        self, th1: float, th2: float, th3: float
    ) -> np.ndarray:
        """
        Compute the gravity vector G(q) = ∂V/∂q.

        V = α1·g·cos θ1 + α2·g·cos θ2 + α3·g·cos θ3
        Differentiating: ∂V/∂θi = −αi·g·sin θi

        Physical interpretation:
          • When θi = 0 (upright) → G = 0 (no gravitational torque — balanced).
          • When θi > 0 (leaning right) → G < 0 → rhs of EOM gets +αi·g·sin θi
            → angular acceleration is positive → link falls further right.
          That is exactly the instability: gravity makes a small lean worse.
        """
        g = self.params.g
        return np.array([
            0.0,                              # x coordinate: gravity is vertical
            -self._alpha1 * g * np.sin(th1),  # link 1 torque from gravity
            -self._alpha2 * g * np.sin(th2),  # link 2 torque from gravity
            -self._alpha3 * g * np.sin(th3),  # link 3 torque from gravity
        ])

    # ── Public interface ──────────────────────────────────────────────────────

    def equations_of_motion(
        self, state: np.ndarray, F_cart: float
    ) -> np.ndarray:
        """
        Compute state derivatives [q̇, q̈] for the ODE integrator.

        We solve the linear system:
            M · q̈  =  Q − h − G − D

        for the four accelerations q̈ = [ẍ, θ̈1, θ̈2, θ̈3].

        Parameters
        ----------
        state   : [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3]  (8-element array)
        F_cart  : External force applied to the cart [N] (positive = rightward)

        Returns
        -------
        dstate  : [ẋ, θ̇1, θ̇2, θ̇3, ẍ, θ̈1, θ̈2, θ̈3]  (8-element array)
                  This is what the ODE integrator needs:
                  "the derivative of the state vector".
        """
        p = self.params
        x, th1, th2, th3, dx, dth1, dth2, dth3 = state

        # ── Build each term of the EOM ────────────────────────────────────
        M = self._mass_matrix(th1, th2, th3)
        h = self._coriolis_vector(th1, th2, th3, dth1, dth2, dth3)
        G = self._gravity_vector(th1, th2, th3)

        # Viscous damping: opposes motion, proportional to speed
        #   Cart: rolling resistance on the rail.
        #   Pivots: friction in the bearings / air resistance on links.
        D = np.array([
            p.cart_friction * dx,    # [N]    — opposes cart sliding
            p.joint_damping * dth1,  # [N·m]  — opposes link 1 rotation
            p.joint_damping * dth2,
            p.joint_damping * dth3,
        ])

        # External generalised force: motor force only acts on the cart (x axis)
        Q = np.array([F_cart, 0.0, 0.0, 0.0])

        # Solve  M · q̈ = Q − h − G − D  (standard NumPy linear solver)
        rhs = Q - h - G - D
        try:
            q_ddot = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            # Degenerate (near-singular) configuration — use least-squares
            q_ddot, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

        # Hard wall: one-sided constraint — the wall provides a normal force
        # only in the inward direction so that friction still decelerates the
        # cart after it has reached the limit.
        if x <= p.x_min and dx < 0:
            q_ddot[0] = max(q_ddot[0], 0.0)   # left wall: only push rightward
        elif x >= p.x_max and dx > 0:
            q_ddot[0] = min(q_ddot[0], 0.0)   # right wall: only push leftward

        # Return full state derivative: [q̇, q̈]
        return np.concatenate([state[4:], q_ddot])

    def mass_matrix(self, state: np.ndarray) -> np.ndarray:
        """
        Return the 4×4 symmetric mass matrix M(θ1, θ2, θ3) for the current
        state.  Column 0 of M⁻¹ gives the sensitivity of each generalised
        acceleration to the cart force, used by SwingUpController.
        """
        _, th1, th2, th3, _, _, _, _ = state
        return self._mass_matrix(th1, th2, th3)

    def total_energy(self, state: np.ndarray) -> float:
        """
        Total mechanical energy of the system: kinetic + potential.

            E = ½·q̇ᵀ·M·q̇  +  Σᵢ mᵢ·g·yᵢ_COM

        where yᵢ_COM is the height of link i's centre of mass above the cart
        pivot (positive upward).

        At the upright equilibrium (all θ=0, all θ̇=0):
            E* = g · [m₁·l₁/2 + m₂·(l₁+l₂/2) + m₃·(l₁+l₂+l₃/2)]

        This is the TARGET energy for the swing-up controller.
        """
        x, th1, th2, th3, dx, dth1, dth2, dth3 = state
        p = self.params

        # Kinetic energy via the full mass matrix (includes all coupling)
        qdot = np.array([dx, dth1, dth2, dth3])
        M    = self._mass_matrix(th1, th2, th3)
        KE   = 0.5 * qdot @ M @ qdot

        # Potential energy (heights relative to cart pivot plane)
        c1, c2, c3 = np.cos(th1), np.cos(th2), np.cos(th3)
        y_COM1 = (p.l1 / 2) * c1
        y_COM2 = p.l1 * c1 + (p.l2 / 2) * c2
        y_COM3 = p.l1 * c1 + p.l2 * c2 + (p.l3 / 2) * c3
        PE = p.g * (p.m1 * y_COM1 + p.m2 * y_COM2 + p.m3 * y_COM3)

        return float(KE + PE)

    def upright_energy(self) -> float:
        """
        Target (upright) total mechanical energy: potential energy when all
        links are exactly vertical (θᵢ=0) and at rest.
        """
        p = self.params
        return float(p.g * (
            p.m1 * (p.l1 / 2)
            + p.m2 * (p.l1 + p.l2 / 2)
            + p.m3 * (p.l1 + p.l2 + p.l3 / 2)
        ))

    def tip_positions(self, state: np.ndarray) -> dict:
        """
        Return Cartesian (x, y) positions of cart and each link tip.
        Used for drawing the animation frame.

        Link tips are computed by walking up the chain:
          tip1 = cart_pivot + l1 · (sin θ1, cos θ1)
          tip2 = tip1       + l2 · (sin θ2, cos θ2)
          tip3 = tip2       + l3 · (sin θ3, cos θ3)
        """
        p = self.params
        x, th1, th2, th3 = state[:4]

        pivot0 = np.array([x, 0.0])
        tip1   = pivot0 + np.array([p.l1 * np.sin(th1),  p.l1 * np.cos(th1)])
        tip2   = tip1   + np.array([p.l2 * np.sin(th2),  p.l2 * np.cos(th2)])
        tip3   = tip2   + np.array([p.l3 * np.sin(th3),  p.l3 * np.cos(th3)])

        return {"cart": pivot0, "tip1": tip1, "tip2": tip2, "tip3": tip3}

    def linearise(self) -> tuple:
        """
        Return the linearised state-space matrices (A, B) at the upright
        equilibrium  q = [0, 0, 0, 0],  q̇ = [0, 0, 0, 0].

        The linearised system is:
            ż = A·z + B·u
        where z = [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3] and u = F_cart.

        A is 8×8, B is 8×1.

        Derivation
        ----------
        At the upright equilibrium, all angles are zero, so:
          cos(θi) = 1, sin(θi) = 0  →  M simplifies to constant M0.
          h = 0  (Coriolis/centrifugal terms are all zero)
          G[i] ≈ −αi·g·θi  (linear approximation of sin θ ≈ θ)

        The linearised EOM is:
          M0 · q̈ = [F, 0, 0, 0]ᵀ − G_lin · q
        where G_lin = diag(0, −α1·g, −α2·g, −α3·g).

        Rearranging into state-space form:
          A = [  0₄    I₄  ]      B = [    0₄   ]
              [ −M0⁻¹·G_lin  0₄ ]      [ M0⁻¹·e₁ ]
        with e₁ = [1, 0, 0, 0]ᵀ (cart-force input column).
        """
        M0   = self._mass_matrix(0.0, 0.0, 0.0)    # mass matrix at upright
        Minv = np.linalg.inv(M0)

        # Linearised gravity stiffness matrix (only diagonal terms survive at θ=0)
        G_lin = np.diag([
            0.0,
            -self._alpha1 * self.params.g,
            -self._alpha2 * self.params.g,
            -self._alpha3 * self.params.g,
        ])

        A = np.zeros((8, 8))
        A[:4, 4:] = np.eye(4)               # q̇ = q̇ (velocities integrate positions)
        A[4:, :4] = -Minv @ G_lin           # q̈ from gravity (sign: EOM has -G_lin*q on rhs)

        B = np.zeros((8, 1))
        B[4:, 0] = Minv[:, 0]              # cart force enters through first column of M⁻¹

        return A, B

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

    For the P and D terms the controller is a *pure function of the current
    state*, which means it can safely be evaluated inside an ODE solver without
    causing numerical issues due to internal history:

      F_cart  = kp_x·(x_ref − x)  −  kd_x·ẋ
              + Σᵢ [ −kp_thᵢ·θᵢ  −  kd_thᵢ·θ̇ᵢ ]

    The velocity ẋ and θ̇ᵢ come directly from the ODE state — no numerical
    differentiation needed.  The D gains damp oscillations that would otherwise
    grow (damping is equivalent to friction added by the control law).

    An optional integral term (ki_x, ki_th) accumulates the angle/position
    error over time and is updated once per control step by the Simulation
    class (not inside the ODE function), so it remains well-defined.

    All gains default to zero — set only the ones you need.

    How to use:
      ctrl = PIDController(
          kp_x=50,  kd_x=20,         # Cart position control
          kp_th=[200, 150, 80],       # Angle proportional gains
          kd_th=[40,  30,  15],       # Angle derivative gains (use velocity)
          ki_th=[1.0, 0.5, 0.2],      # Angle integral gains (small!)
          max_force=300,
      )
      force = ctrl.compute(state, dt=0.01, x_ref=0.0)
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
    ) -> None:
        """
        Parameters
        ----------
        kp_x, ki_x, kd_x : PID gains for cart x-position.
        kp_th             : Proportional gains for [θ1, θ2, θ3].
        ki_th             : Integral gains   for [θ1, θ2, θ3] (keep small).
        kd_th             : Derivative gains for [θ1, θ2, θ3].
        max_force         : Clamp on total output force [N].
        """
        self.kp_x = kp_x
        self.ki_x = ki_x
        self.kd_x = kd_x
        self.kp_th = list(kp_th or [0.0, 0.0, 0.0])
        self.ki_th = list(ki_th or [0.0, 0.0, 0.0])
        self.kd_th = list(kd_th or [0.0, 0.0, 0.0])
        self.max_force = max_force

        # Integral accumulators — updated once per control step, not inside ODE
        self._integral_x: float = 0.0
        self._integral_th: List[float] = [0.0, 0.0, 0.0]

    def reset(self) -> None:
        """Reset integral accumulators to zero."""
        self._integral_x = 0.0
        self._integral_th = [0.0, 0.0, 0.0]

    def update_integrals(self, state: np.ndarray, dt: float, x_ref: float = 0.0) -> None:
        """
        Advance the integral terms by one control step.

        Called by the Simulation once per control interval (not inside the ODE),
        so each control step accumulates exactly one dt of integral — regardless
        of how many ODE sub-steps the solver takes internally.
        """
        x, th1, th2, th3 = state[:4]
        self._integral_x += (x_ref - x) * dt   # ∫(x_ref − x) dt → positive when left of target
        for i, th in enumerate([th1, th2, th3]):
            self._integral_th[i] += th * dt      # ∫θ dt → positive when leaning right

    def compute(
        self, state: np.ndarray, dt: float = 0.01, x_ref: float = 0.0
    ) -> float:
        """
        Compute the control force for the current state.

        The P and D terms are pure functions of the state vector.
        The I terms use the accumulators built up by update_integrals().

        Parameters
        ----------
        state : [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3]
        dt    : Control step size [s] — used only to advance integrals when
                update_integrals() has not been called separately.
        x_ref : Desired cart position [m] (default 0.0).

        Returns
        -------
        force : Control force to apply to the cart [N].
        """
        x, th1, th2, th3, dx, dth1, dth2, dth3 = state

        # ── Cart position PD + I ─────────────────────────────────────────────
        err_x = x_ref - x
        # P: proportional to position error (spring-like restoring force)
        # D: proportional to cart velocity (adds electronic damping to the cart)
        # I: accumulated position error (removes steady-state offset)
        f_x = (
            self.kp_x * err_x
            - self.kd_x * dx            # ẋ already is d(x)/dt → D term
            + self.ki_x * self._integral_x
        )

        # ── Angle PD + I for each link ────────────────────────────────────────
        # SIGN CONVENTION (critical):
        #   If θ > 0 (pendulum leans RIGHT), push cart RIGHT (+F).
        #   This causes cart acceleration to the right → pendulum lags behind
        #   by inertia → pendulum tips LEFT relative to cart → corrects lean.
        #   So: F_th = +kp·θ + kd·θ̇  (POSITIVE gains, POSITIVE angle = POSITIVE force)
        f_th = 0.0
        for i, (th, dth, kp, ki, kd) in enumerate(
            zip([th1, th2, th3], [dth1, dth2, dth3],
                self.kp_th, self.ki_th, self.kd_th)
        ):
            # P: push cart in direction of lean (+kp·θ)
            # D: additional push if pendulum is falling faster (+kd·θ̇)
            # I: correct any persistent lean (+ki · ∫θ dt)
            f_th += (
                kp * th
                + kd * dth
                + ki * self._integral_th[i]
            )

        total = f_x + f_th
        return float(np.clip(total, -self.max_force, self.max_force))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4b — LQRController
# ══════════════════════════════════════════════════════════════════════════════

class LQRController:
    """
    Linear Quadratic Regulator (LQR) controller for the triple-pendulum cart.

    --- WHAT IS LQR? ---

    LQR is an *optimal* state-feedback controller that minimises the cost:
        J = ∫₀^∞  ( zᵀ·Q·z  +  u·R·u )  dt

    where:
      z = full state vector  [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3]
      u = cart force  F_cart
      Q = 8×8 diagonal matrix — penalises state deviation (larger = care more)
      R = scalar — penalises control effort  (larger = gentler control)

    The optimal control law is *linear* in the state:
        F_cart = −K · z

    where K (the 1×8 gain matrix) is computed by solving the algebraic
    Riccati equation:
        Aᵀ·P + P·A − P·B·R⁻¹·Bᵀ·P + Q = 0
    and then:
        K = R⁻¹ · Bᵀ · P

    A and B come from linearising the physics about the upright equilibrium
    (see TriplePendulumPhysics.linearise()).

    --- WHY DOES THIS WORK? ---

    By weighting the angles heavily in Q (e.g. 1000× the cart position),
    the optimiser finds gains that strongly damp angle deviations while
    keeping control effort reasonable. The resulting K automatically couples
    all 8 state variables — the angle D-terms come "for free" from the
    optimal solution without us having to tune them manually.

    How to modify:
      • Q_diag — list of 8 values [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3].
        Increase Q_diag[0] to penalise cart drift more.
        Increase Q_diag[1..3] to more aggressively correct link angles.
      • R — increase to produce a gentler, lower-force controller (may
        not stabilise if gains are too small).
      • max_force — hard clamp on the output [N].
    """

    def __init__(
        self,
        physics: "TriplePendulumPhysics",
        Q_diag: Optional[List[float]] = None,
        R: float = 0.001,
        max_force: float = 600.0,
    ) -> None:
        """
        Parameters
        ----------
        physics   : TriplePendulumPhysics instance (used for linearisation).
        Q_diag    : 8-element list of diagonal Q-matrix weights.
                    Default: heavily penalise angles, lightly penalise cart pos.
        R         : Control-effort weight scalar.  Smaller → more aggressive.
        max_force : Saturation limit on output force [N].
        """
        from scipy.linalg import solve_continuous_are

        self.max_force = max_force

        # ── Default Q weights ────────────────────────────────────────────────
        # State:  [x,    θ1,    θ2,    θ3,   ẋ,    θ̇1,   θ̇2,   θ̇3]
        if Q_diag is None:
            Q_diag = [10.0, 1000.0, 1000.0, 1000.0, 1.0, 50.0, 50.0, 50.0]

        Q = np.diag(Q_diag)

        # ── Linearise at upright equilibrium ─────────────────────────────────
        A, B = physics.linearise()

        # ── Solve the algebraic Riccati equation ─────────────────────────────
        # Aᵀ P + P A − P B R⁻¹ Bᵀ P + Q = 0
        try:
            P = solve_continuous_are(A, B, Q, np.array([[R]]))
        except Exception as e:
            raise RuntimeError(
                f"LQR Riccati equation failed: {e}\n"
                "Check that the system is controllable (A, B) and Q is positive semi-definite."
            ) from e

        # ── Optimal gain matrix K ─────────────────────────────────────────────
        # K = R⁻¹ · Bᵀ · P    (shape 1×8)
        self.K = (1.0 / R) * B.T @ P

        # ── Print eigenvalues of the closed-loop system A−BK ─────────────────
        A_cl = A - B @ self.K
        eigs = np.linalg.eigvals(A_cl)
        eigs_sorted = sorted(eigs, key=lambda e: e.real)
        print("  LQR closed-loop eigenvalues (all should have Re<0 for stability):")
        for ev in eigs_sorted:
            stability = "✓" if ev.real < 0 else "✗ UNSTABLE"
            print(f"    {ev.real:+.3f} ± {abs(ev.imag):.3f}j  {stability}")

    def reset(self) -> None:
        """LQR has no internal state to reset."""

    def update_integrals(self, state: np.ndarray, dt: float, x_ref: float = 0.0) -> None:
        """LQR has no integral state — this is a no-op."""

    def compute(
        self, state: np.ndarray, dt: float = 0.01, x_ref: float = 0.0
    ) -> float:
        """
        Compute the optimal LQR control force.

        F = −K · (state − reference)

        Since the reference is the upright equilibrium with x = x_ref,
        we subtract x_ref from the cart position before applying K.

        Parameters
        ----------
        state : [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3]
        dt    : Unused — LQR is a purely static function of state.
        x_ref : Desired cart position [m].

        Returns
        -------
        force : Optimal control force [N].
        """
        z = state.copy()
        z[0] -= x_ref   # express cart position relative to reference
        F = -float((self.K @ z)[0])
        return float(np.clip(F, -self.max_force, self.max_force))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4c — SwingUpController
# ══════════════════════════════════════════════════════════════════════════════

class SwingUpController:
    """
    Energy-based swing-up controller for the triple inverted pendulum.

    Uses the Åström–Furuta energy-pumping method: the cart is driven to inject
    mechanical energy into each pendulum link, swinging them from the stable
    hanging equilibrium (θ = π) to the unstable upright (θ = 0).

    The Lyapunov function for the TOTAL system energy

        V = (E_total − E*)²,   E* = upright potential energy (at rest)

    decreases monotonically under the control law

        F = −k_energy · ΔE_total · ẋ_cart,   ΔE_total = E_total − E*

    because  dE_total/dt = F_cart · ẋ_cart  (external power input), giving

        V̇ = 2·ΔE · Ė = 2·ΔE · (−k·ΔE·ẋ²) = −2·k·ΔE²·ẋ² ≤ 0  ✓

    Physical interpretation:
      • When the total energy is below the upright target (ΔE < 0) and the
        cart is moving rightward (ẋ > 0): the force is positive (rightward),
        ADDING energy to the system → oscillation amplitude grows.
      • When the total energy overshoots the target (ΔE > 0): the force sign
        reverses → excess energy is removed (braking).
    This correctly handles ALL three links simultaneously because the formula
    targets the FULL kinetic + potential energy of the coupled system.

    A strong cart-centering spring prevents the cart from reaching the rail
    ends while the pump is active.

    Switch condition (call is_near_upright()):
        All |normalise(θᵢ)| < switch_angle  →  hand off to LQR controller.
    """

    def __init__(
        self,
        physics: "TriplePendulumPhysics",
        k_energy: float = 50.0,
        max_force: float = 300.0,
        switch_angle_deg: float = 20.0,
        cart_k: float = 100.0,
        cart_d: float = 20.0,
    ) -> None:
        """
        Parameters
        ----------
        physics          : TriplePendulumPhysics (for physical constants).
        k_energy         : Energy-pumping gain [N/J].  The formula is
                           F_pump = −k_energy · ΔE_total · ẋ_cart.
        max_force        : Saturation limit on the output force [N].
        switch_angle_deg : Angle threshold for LQR handoff [°].
                           All three links must be within this of upright.
        cart_k           : Soft spring gain for cart centering [N/m].
        cart_d           : Soft damper gain for cart centering [N·s/m].
        """
        self.physics      = physics
        self.k_energy     = k_energy
        self.max_force    = max_force
        self.switch_angle = np.radians(switch_angle_deg)
        self.cart_k       = cart_k
        self.cart_d       = cart_d
        self._p           = physics.params
        # Cache the target energy (constant for a given system)
        self._E_star      = physics.upright_energy()

    def is_near_upright(self, state: np.ndarray) -> bool:
        """
        Return True when all three links are within switch_angle of the upright.

        Angles are normalised to (−π, π] before comparison so that wrapped
        angles (e.g. θ > 2π after a full revolution) are handled correctly.
        """
        for theta in state[1:4]:
            theta_norm = (theta + np.pi) % (2 * np.pi) - np.pi
            if abs(theta_norm) > self.switch_angle:
                return False
        return True

    def reset(self) -> None:
        """No internal state to reset."""

    def update_integrals(
        self, state: np.ndarray, dt: float, x_ref: float = 0.0
    ) -> None:
        """No integral state — no-op."""

    def compute(
        self, state: np.ndarray, dt: float = 0.01, x_ref: float = 0.0
    ) -> float:
        """
        Compute the energy-pumping control force.

        Uses the TOTAL system energy (kinetic + potential of all links and
        cart) rather than individual per-link energies.  The Lyapunov function

            V = (E_total − E*)²

        decreases under the control law

            F = −k_energy · ΔE_total · ẋ_cart

        where ΔE_total = E_total − E* (negative when energy is below target).

        Derivation: dE_total/dt = F_cart · ẋ_cart (external power = F·v).
        Choosing F = −k · ΔE · ẋ  gives  V̇ = −2k · ΔE² · ẋ² ≤ 0. ✓

        A wall-proximity factor suppresses the pumping near the rail ends so
        the centering term can bring the cart back into bounds.

        Parameters
        ----------
        state : [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3]
        dt    : Unused (stateless controller).
        x_ref : Desired cart centre position [m].

        Returns
        -------
        force : Control force [N], clamped to ±max_force.
        """
        x, th1, th2, th3, dx, dth1, dth2, dth3 = state
        p = self._p

        # Total-energy error: negative when below upright target
        dE = self.physics.total_energy(state) - self._E_star

        # Lyapunov-based pumping: F = −k · ΔE · ẋ
        F_pump = -self.k_energy * dE * dx

        # Wall-proximity factor: smoothly suppress pumping near rail ends
        x_half = (p.x_max - p.x_min) / 2.0
        if x_half > 0:
            x_norm     = (x - x_ref) / x_half       # ∈ [−1, +1]
            wall_factor = max(0.0, 1.0 - x_norm ** 4)
        else:
            wall_factor = 1.0
        F_pump *= wall_factor

        # Strong restoring force keeps the cart near the centre of the rail
        F_centre = -self.cart_k * (x - x_ref) - self.cart_d * dx

        total = F_pump + F_centre
        return float(np.clip(total, -self.max_force, self.max_force))


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

    Integration strategy — Zero-Order Hold (ZOH)
    ---------------------------------------------
    We split simulation time into discrete control intervals of length
    dt_output.  At the START of each interval:
      1. The controller evaluates the current state → produces a force F.
      2. F is held constant for the whole interval.
      3. scipy.integrate.solve_ivp integrates the ODE (with constant F)
         from t_now to t_now + dt_output using the RK45 adaptive method.
      4. The controller's integral accumulators are updated by dt_output.

    Because the ODE function is a pure function of (t, state) within each
    interval (F is constant, no controller state is modified during ODE
    evaluation), the adaptive step-size solver works correctly.

    How to run a simulation:
      sim = Simulation(physics, controller, motor)
      result = sim.run(
          t_span=(0, 5),       # Simulate 5 seconds
          initial_state=...,   # 8-element array
          dt_output=0.01,      # Control step + output resolution [s]
      )
    """

    def __init__(
        self,
        physics: TriplePendulumPhysics,
        controller=None,    # PIDController, LQRController, or None
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
        Integrate the system forward in time using Zero-Order Hold control.

        Parameters
        ----------
        t_span       : (t_start, t_end) in seconds.
        initial_state: [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3]
        dt_output    : Control step size AND output resolution [s].
        use_motor_model : If True, pass controller command through MotorModel
                         (adds back-EMF / saturation effects).

        Returns
        -------
        SimulationResult with all time series.
        """
        if self.controller:
            self.controller.reset()

        # Stall force of the motor — used to normalise controller output to
        # a voltage command in [−1, +1] before feeding the motor model.
        stall_f = self.motor.stall_force() if self.motor else 1.0

        t_start, t_end = t_span
        t_steps = np.arange(t_start, t_end, dt_output)

        # Pre-allocate output arrays
        n = len(t_steps)
        out_states = np.empty((n, 8))
        out_forces = np.empty(n)

        state = np.array(initial_state, dtype=float)
        print(f"Integrating t=[{t_start}, {t_end}] s  ({n} control steps) …")

        for k, t_now in enumerate(t_steps):
            t_next = min(t_now + dt_output, t_end)

            # ── 1. Evaluate control force at current state ─────────────────
            if self.controller is not None:
                f_cmd = self.controller.compute(state, dt_output)
            else:
                f_cmd = 0.0

            # ── 2. Pass through motor model (back-EMF, saturation) ─────────
            if use_motor_model and self.motor is not None:
                # Normalise desired force to a voltage command [−1, +1]
                command = float(np.clip(f_cmd / stall_f, -1.0, 1.0))
                F_cart  = self.motor.force_from_command(command, cart_speed_ms=state[4])
            else:
                F_cart = f_cmd

            # Record output for this step
            out_states[k] = state
            out_forces[k] = F_cart

            # ── 3. Integrate ODE for one control interval (F is constant) ──
            # The ODE function is now stateless within this interval.
            def ode_rhs(t: float, s: np.ndarray, _F: float = F_cart) -> np.ndarray:
                return self.physics.equations_of_motion(s, _F)

            sol = solve_ivp(
                ode_rhs,
                [t_now, t_next],
                state,
                method="RK45",
                rtol=1e-6,
                atol=1e-8,
                dense_output=False,
            )

            if not sol.success:
                print(f"  Warning: ODE solver issue at t={t_now:.3f}: {sol.message}")
                # Keep the last successfully integrated state and continue
            else:
                state = sol.y[:, -1]

            # ── 4. Update controller integral accumulators ─────────────────
            # Done AFTER the ODE step (so integrals advance once per control
            # interval, not once per ODE sub-step).
            if self.controller is not None:
                self.controller.update_integrals(state, dt_output)

        print(f"  Integration complete. {n} output points.")
        return SimulationResult(
            t=t_steps,
            x=out_states[:, 0],
            theta1=out_states[:, 1],
            theta2=out_states[:, 2],
            theta3=out_states[:, 3],
            dx=out_states[:, 4],
            dtheta1=out_states[:, 5],
            dtheta2=out_states[:, 6],
            dtheta3=out_states[:, 7],
            force=out_forces,
            states=out_states,
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

# Each config function returns a 5-tuple:
#   (SystemParameters, MotorModel, PIDController | None, initial_state, use_motor_model)
# use_motor_model=False → apply the controller force directly (ideal actuator),
#                         useful for demonstrating pure control theory.
# use_motor_model=True  → route the force through the DC motor + lead-screw
#                         model (realistic back-EMF and speed limits).

def config_free_swing() -> tuple:
    """
    Configuration A — Free swing (no control).

    All three links start from a nearly-upright position with a small
    perturbation.  No control force is applied, so the pendulum falls freely.
    Use this to verify that the physics are correct:
      • Energy should be approximately conserved (only small damping).
      • The angles should grow continuously — the system is unstable.
    """
    params = SystemParameters(
        M_cart=2.0,
        l1=0.30, m1=0.15,
        l2=0.25, m2=0.10,
        l3=0.20, m3=0.08,
        cart_friction=0.5,    # Low friction to preserve energy
        joint_damping=0.0001,
    )
    motor = MotorModel()
    controller = None

    # Start perfectly upright with a tiny nudge on each link
    initial_state = np.array([
        0.0,           # x  [m]
        np.radians(3), # θ1 [rad] — 3° lean on bottom link
        np.radians(2), # θ2
        np.radians(1), # θ3
        0.0, 0.0, 0.0, 0.0,  # all velocities zero
    ])
    return params, motor, controller, initial_state, False


def config_pd_stabilise() -> tuple:
    """
    Configuration B — LQR stabilisation near the upright (ideal actuator).

    Uses an LQR (Linear Quadratic Regulator) controller — the standard
    engineering solution for stabilising unstable linear systems. LQR
    automatically computes the optimal feedback gains by solving the
    algebraic Riccati equation on the linearised plant model.

    Why not plain PD?
    -----------------
    A triple inverted pendulum has 4 coupled unstable modes. Choosing PD
    gains independently for each link ignores the cross-coupling between
    them, making it very easy to accidentally pick gains that are either
    too weak (pendulum falls) or violate the closed-loop stability
    condition.  LQR avoids this by finding the globally optimal gains for
    the Q/R weights you specify — it is much harder to mis-tune.

    The motor model is bypassed (use_motor_model=False) so we demonstrate
    the pure control theory without the realistic motor speed limit.
    See config_secondary_motor() for the same with the full motor model.

    How to tune:
      Q_diag[1..3] — higher → correct angles more aggressively.
      Q_diag[0]    — higher → keep cart closer to x=0.
      R            — smaller → more aggressive force (may saturate).
    """
    params = SystemParameters(
        M_cart=2.0,
        l1=0.30, m1=0.15,
        l2=0.25, m2=0.10,
        l3=0.20, m3=0.08,
    )
    motor = MotorModel()   # Motor present but bypassed for this config
    physics = TriplePendulumPhysics(params)

    print("Computing LQR gains (pd_stabilise)…")
    controller = LQRController(
        physics=physics,
        # State weights: [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3]
        Q_diag=[10.0, 2000.0, 2000.0, 2000.0, 1.0, 100.0, 100.0, 100.0],
        R=0.001,        # Low R → controller is allowed to use large forces
        max_force=600.0,
    )

    # Small perturbation — LQR is a *local* controller (only works near upright)
    initial_state = np.array([
        0.0,
        np.radians(3),    # θ1: 3° lean
        np.radians(2),    # θ2: 2° lean
        np.radians(1.5),  # θ3: 1.5° lean
        0.0, 0.0, 0.0, 0.0,
    ])
    # use_motor_model=False → force applied directly (ideal actuator demo)
    return params, motor, controller, initial_state, False


def config_longer_links() -> tuple:
    """
    Configuration C — Longer, heavier links with realistic motor model.

    Tests the simulation with a different link geometry through the full
    motor model (1120 RPM, 8 mm lead).  Longer links have:
      • More rotational inertia (harder to move quickly)
      • Slower instability (lower ωn = √(g*α/I)) — actually EASIER to stabilise
      • Larger gravitational torques (need more force)

    The LQR controller adapts its gains automatically to the new link
    parameters — just change the geometry and re-run.
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
    physics = TriplePendulumPhysics(params)

    print("Computing LQR gains (longer_links)…")
    controller = LQRController(
        physics=physics,
        Q_diag=[10.0, 2000.0, 2000.0, 2000.0, 1.0, 100.0, 100.0, 100.0],
        R=0.001,
        max_force=400.0,
    )
    initial_state = np.array([
        0.0,
        np.radians(3),
        np.radians(2),
        np.radians(1),
        0.0, 0.0, 0.0, 0.0,
    ])
    # Motor model is bypassed (ideal force) so LQR can demonstrate purely
    # the effect of changing link geometry on the stabilisation response.
    # With the 8 mm / 1120 RPM lead-screw model active (True), the cart
    # speed cap of 0.149 m/s prevents the LQR from delivering the required
    # forces fast enough and the pendulum will fall — this is physically
    # realistic but less instructive for a geometry comparison demo.
    return params, motor, controller, initial_state, False


def config_secondary_motor() -> tuple:
    """
    Configuration D — LQR controller, heavy links, secondary 312 RPM motor.

    Demonstrates the effect of heavier links and a slower (but stronger)
    motor on the stabilisation behaviour.  The motor model is bypassed so
    the ideal LQR force is applied directly — this lets you compare the
    settling behaviour cleanly without motor-bandwidth artefacts.

    To enable the realistic motor model and see its force-limiting effects,
    change the return value below from False → True and observe that the
    pendulum falls once the cart speed exceeds ~0.04 m/s (the 312 RPM
    motor's free-run limit with an 8 mm lead screw).
    """
    params = SystemParameters(
        M_cart=4.0,
        l1=0.30, m1=0.30,
        l2=0.25, m2=0.25,
        l3=0.20, m3=0.20,
        cart_friction=8.0,
    )
    motor = MotorModel.secondary_motor()
    physics = TriplePendulumPhysics(params)

    print("Computing LQR gains (secondary_motor)…")
    controller = LQRController(
        physics=physics,
        Q_diag=[10.0, 2000.0, 2000.0, 2000.0, 1.0, 100.0, 100.0, 100.0],
        R=0.001,
        max_force=600.0,
    )
    initial_state = np.array([
        0.0,
        np.radians(3),
        np.radians(2),
        np.radians(1.5),
        0.0, 0.0, 0.0, 0.0,
    ])
    # Motor model bypassed: ideal force so LQR stabilises cleanly.
    # Flip to True to see the motor speed limit cause the pendulum to fall.
    return params, motor, controller, initial_state, False


def config_swing_up() -> tuple:
    """
    Configuration E — Swing-up from hanging to balanced (ideal actuator).

    All three links start hanging straight down (θ ≈ π) with a tiny
    perturbation to break exact symmetry.  A SwingUpController uses the
    energy-pumping method to swing the links up toward the upright position.

    This configuration is used automatically by run_interactive() and can
    also be run non-interactively for a quick end-to-end test.  When running
    interactively the simulation pauses at the hanging state until the user
    presses Start.
    """
    params = SystemParameters(
        M_cart=2.0,
        l1=0.30, m1=0.15,
        l2=0.25, m2=0.10,
        l3=0.20, m3=0.08,
    )
    motor = MotorModel()
    physics = TriplePendulumPhysics(params)

    controller = SwingUpController(
        physics=physics,
        k_energy=50.0,
        max_force=300.0,
        switch_angle_deg=20.0,
        cart_k=100.0,
        cart_d=20.0,
    )

    # Hanging down with a tiny perturbation so the symmetry is broken
    initial_state = np.array([
        0.0,
        np.pi - np.radians(2.0),   # θ1: 2° off the hanging vertical
        np.pi - np.radians(1.0),   # θ2: 1° off
        np.pi - np.radians(0.5),   # θ3: 0.5° off
        0.0, 0.0, 0.0, 0.0,
    ])
    return params, motor, controller, initial_state, False


# Map of named configurations for easy selection on the command line
CONFIGURATIONS = {
    "free_swing":       config_free_swing,
    "pd_stabilise":     config_pd_stabilise,
    "longer_links":     config_longer_links,
    "secondary_motor":  config_secondary_motor,
    "swing_up":         config_swing_up,
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — run_interactive() and main()
# ══════════════════════════════════════════════════════════════════════════════

def run_interactive(
    dt: float = 0.01,
    show_plots: bool = True,
) -> "animation.FuncAnimation":
    """
    Interactive real-time simulation with Start and Reset buttons.

    The triple pendulum is shown in the hanging-down position (θ = π for all
    links) when the window opens.  Click **▶ Start** to begin the energy-based
    swing-up sequence.  The simulation automatically switches to the LQR
    balancing controller once all three links are within 20° of upright.
    Click **↺ Reset** at any time to stop the simulation and return to the
    hanging start state.

    Two live plots on the right side of the window show the current link
    angles and the applied control force as the simulation runs.

    Parameters
    ----------
    dt         : Control / animation time step [s] (default 0.01 → 100 Hz).
    show_plots : If True, call plt.show() (blocks until the window is closed).

    Returns
    -------
    anim : The FuncAnimation object (kept alive by the figure).
    """
    from matplotlib.widgets import Button as MplButton

    # ── Physics and controllers ────────────────────────────────────────────────
    params = SystemParameters(
        M_cart=2.0,
        l1=0.30, m1=0.15,
        l2=0.25, m2=0.10,
        l3=0.20, m3=0.08,
    )
    physics = TriplePendulumPhysics(params)
    p = params

    print("Computing LQR gains for balancing phase…")
    lqr = LQRController(
        physics=physics,
        Q_diag=[10.0, 2000.0, 2000.0, 2000.0, 1.0, 100.0, 100.0, 100.0],
        R=0.001,
        max_force=600.0,
    )
    swing_up = SwingUpController(
        physics=physics,
        k_energy=50.0,
        max_force=300.0,
        switch_angle_deg=20.0,
        cart_k=100.0,
        cart_d=20.0,
    )

    # ── Initial state: all links hanging straight down ─────────────────────────
    _INITIAL = np.array([
        0.0,
        np.pi - np.radians(2.0),   # θ1: 2° off hanging vertical (breaks symmetry)
        np.pi - np.radians(1.0),   # θ2: 1° off
        np.pi - np.radians(0.5),   # θ3: 0.5° off
        0.0, 0.0, 0.0, 0.0,        # all velocities zero
    ])

    # ── Mutable simulation state (lists so nested functions can write to them) ─
    state    = [_INITIAL.copy()]   # [0] = current 8-element state vector
    running  = [False]             # [0] = True while simulation is active
    phase    = ["idle"]            # [0] = 'idle' | 'swinging_up' | 'balancing'
    sim_time = [0.0]               # [0] = elapsed simulation time [s]

    t_history  : List[float] = []
    th1_history: List[float] = []
    th2_history: List[float] = []
    th3_history: List[float] = []
    F_history  : List[float] = []
    MAX_HIST = 500  # rolling-window length (frames kept in the live plots)

    # ── Figure layout ──────────────────────────────────────────────────────────
    total_len = p.l1 + p.l2 + p.l3
    margin    = 0.15

    fig = plt.figure(figsize=(13, 7))
    fig.suptitle(
        "Triple Inverted Pendulum — Interactive Simulation\n"
        "Start hanging ↓  →  swing-up  →  balance ↑",
        fontsize=11,
    )

    # Left panel: pendulum animation
    ax_anim = fig.add_axes([0.04, 0.13, 0.52, 0.80])
    ax_anim.set_xlim(p.x_min - margin, p.x_max + margin)
    ax_anim.set_ylim(-total_len - margin, total_len + margin)
    ax_anim.set_aspect("equal")
    ax_anim.set_xlabel("Cart position x [m]")
    ax_anim.set_ylabel("Height [m]")
    ax_anim.axhline(0.0,     color="gray",   linewidth=0.8, linestyle="--")
    ax_anim.axvline(p.x_min, color="salmon", linewidth=1,   linestyle=":")
    ax_anim.axvline(p.x_max, color="salmon", linewidth=1,   linestyle=":")
    ax_anim.plot([p.x_min, p.x_max], [0, 0], "k-", linewidth=4, zorder=1)

    # Cart rectangle and pendulum links
    cart_w, cart_h = 0.12, 0.06
    cart_patch = plt.Rectangle(
        (-cart_w / 2, -cart_h), cart_w, cart_h,
        fc="steelblue", ec="navy", zorder=3,
    )
    ax_anim.add_patch(cart_patch)

    colors = ["tomato", "goldenrod", "mediumseagreen"]
    link_lines = [
        ax_anim.plot([], [], "-o", color=c, linewidth=3, markersize=6, zorder=4)[0]
        for c in colors
    ]
    pivot_dot = ax_anim.plot([], [], "ko", markersize=8, zorder=5)[0]

    # Text overlays
    time_text = ax_anim.text(
        0.02, 0.97, "Press  \u25b6 Start  to begin",
        transform=ax_anim.transAxes, fontsize=10, va="top",
    )
    phase_text = ax_anim.text(
        0.02, 0.91, "",
        transform=ax_anim.transAxes, fontsize=9, va="top", color="royalblue",
    )
    angle_text = ax_anim.text(
        0.70, 0.97, "",
        transform=ax_anim.transAxes, fontsize=9, va="top", family="monospace",
    )

    # Right panel — live angle plot
    ax_th = fig.add_axes([0.60, 0.57, 0.38, 0.35])
    ax_th.set_ylabel("Angle [°]")
    ax_th.set_title("Link Angles (normalised to ±180°)", fontsize=9)
    ax_th.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_th.grid(True, alpha=0.3)
    line_th1, = ax_th.plot([], [], color="tomato",         label="θ1 (bottom)")
    line_th2, = ax_th.plot([], [], color="goldenrod",      label="θ2 (middle)")
    line_th3, = ax_th.plot([], [], color="mediumseagreen", label="θ3 (top)")
    ax_th.legend(fontsize=8, loc="upper right")

    # Right panel — live force plot
    ax_f = fig.add_axes([0.60, 0.13, 0.38, 0.35])
    ax_f.set_ylabel("Force [N]")
    ax_f.set_xlabel("Time [s]")
    ax_f.set_title("Control Force", fontsize=9)
    ax_f.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_f.grid(True, alpha=0.3)
    line_f, = ax_f.plot([], [], color="purple", label="F_cart")
    ax_f.legend(fontsize=8, loc="upper right")

    # ── Buttons ────────────────────────────────────────────────────────────────
    ax_btn_start = fig.add_axes([0.10, 0.02, 0.16, 0.07])
    ax_btn_reset = fig.add_axes([0.30, 0.02, 0.16, 0.07])
    btn_start = MplButton(ax_btn_start, "\u25b6  Start")
    btn_reset = MplButton(ax_btn_reset, "\u21ba  Reset")

    def _start_cb(event):
        if not running[0]:
            running[0] = True
            phase[0] = "swinging_up"
            btn_start.label.set_text("Running\u2026")
            fig.canvas.draw_idle()

    def _reset_cb(event):
        state[0] = _INITIAL.copy()
        running[0] = False
        phase[0] = "idle"
        sim_time[0] = 0.0
        t_history.clear()
        th1_history.clear()
        th2_history.clear()
        th3_history.clear()
        F_history.clear()
        btn_start.label.set_text("\u25b6  Start")
        time_text.set_text("Press  \u25b6 Start  to begin")
        phase_text.set_text("")
        angle_text.set_text("")
        line_th1.set_data([], [])
        line_th2.set_data([], [])
        line_th3.set_data([], [])
        line_f.set_data([], [])
        _draw_pendulum(state[0])
        fig.canvas.draw_idle()

    btn_start.on_clicked(_start_cb)
    btn_reset.on_clicked(_reset_cb)

    # ── Pendulum drawing helper ────────────────────────────────────────────────
    def _draw_pendulum(s: np.ndarray) -> None:
        tips   = physics.tip_positions(s)
        cart_x = tips["cart"][0]
        cart_patch.set_xy((cart_x - cart_w / 2, -cart_h))
        pivot_dot.set_data([cart_x], [0.0])
        link_lines[0].set_data(
            [cart_x,         tips["tip1"][0]],
            [0.0,             tips["tip1"][1]],
        )
        link_lines[1].set_data(
            [tips["tip1"][0], tips["tip2"][0]],
            [tips["tip1"][1], tips["tip2"][1]],
        )
        link_lines[2].set_data(
            [tips["tip2"][0], tips["tip3"][0]],
            [tips["tip2"][1], tips["tip3"][1]],
        )

    # Draw the initial hanging state immediately so the window is not blank
    _draw_pendulum(_INITIAL)

    # ── Animation update callback ──────────────────────────────────────────────
    _static_artists = (
        link_lines + [cart_patch, pivot_dot, time_text, phase_text, angle_text]
    )

    def _update(_frame):
        if not running[0]:
            return _static_artists

        s = state[0]

        # Phase-transition logic
        if phase[0] == "swinging_up":
            ctrl = swing_up
            if swing_up.is_near_upright(s):
                phase[0] = "balancing"
                lqr.reset()
        elif phase[0] == "balancing":
            ctrl = lqr
            # Fall back to swing-up if any link drifts too far from upright
            if any(
                abs((th + np.pi) % (2 * np.pi) - np.pi) > np.radians(35)
                for th in s[1:4]
            ):
                phase[0] = "swinging_up"
        else:
            return _static_artists

        # Integrate one control step (constant force over [0, dt])
        F = ctrl.compute(s, dt)

        def _ode(t_local: float, s_local: np.ndarray, _F: float = F) -> np.ndarray:
            return physics.equations_of_motion(s_local, _F)

        sol = solve_ivp(
            _ode, [0.0, dt], s,
            method="RK45", rtol=1e-6, atol=1e-8,
        )
        if sol.success:
            state[0] = sol.y[:, -1]
        sim_time[0] += dt
        s = state[0]

        # Record normalised angles for the live plot
        th1_n = (s[1] + np.pi) % (2 * np.pi) - np.pi
        th2_n = (s[2] + np.pi) % (2 * np.pi) - np.pi
        th3_n = (s[3] + np.pi) % (2 * np.pi) - np.pi

        t_history.append(sim_time[0])
        th1_history.append(np.degrees(th1_n))
        th2_history.append(np.degrees(th2_n))
        th3_history.append(np.degrees(th3_n))
        F_history.append(F)

        # Trim rolling window
        if len(t_history) > MAX_HIST:
            t_history.pop(0)
            th1_history.pop(0)
            th2_history.pop(0)
            th3_history.pop(0)
            F_history.pop(0)

        # Draw pendulum geometry
        _draw_pendulum(s)

        # Update text overlays
        time_text.set_text(f"t = {sim_time[0]:.2f} s")
        phase_labels = {
            "swinging_up": "\u26a1 Swinging up\u2026",
            "balancing":   "\u2713 Balancing",
        }
        phase_text.set_text(phase_labels.get(phase[0], ""))
        angle_text.set_text(
            f"\u03b81 = {np.degrees(th1_n):+6.1f}\u00b0\n"
            f"\u03b82 = {np.degrees(th2_n):+6.1f}\u00b0\n"
            f"\u03b83 = {np.degrees(th3_n):+6.1f}\u00b0"
        )

        # Update live plots
        t_arr = list(t_history)
        line_th1.set_data(t_arr, list(th1_history))
        line_th2.set_data(t_arr, list(th2_history))
        line_th3.set_data(t_arr, list(th3_history))
        ax_th.relim()
        ax_th.autoscale_view()

        line_f.set_data(t_arr, list(F_history))
        ax_f.relim()
        ax_f.autoscale_view()

        return _static_artists

    anim = animation.FuncAnimation(
        fig, _update,
        interval=10,             # ~100 Hz target; actual speed depends on host
        blit=False,              # blit=False required because axes limits change
        cache_frame_data=False,
    )
    # Keep a reference on the figure so the animation is not garbage-collected
    fig._anim_ref = anim  # type: ignore[attr-defined]

    if show_plots:
        plt.show()

    return anim


def main(
    config_name: str = "pd_stabilise",
    t_end: float = 5.0,
    dt_output: float = 0.01,
    animate: bool = True,
    save_animation: Optional[str] = None,
    save_plot: Optional[str] = None,
    show_plots: bool = True,
    interactive: bool = False,
) -> Optional[SimulationResult]:
    """
    Run the triple-inverted-pendulum simulation with a named configuration.

    Parameters
    ----------
    config_name    : One of the keys in CONFIGURATIONS.
    t_end          : Simulation end time [s].
    dt_output      : Control step size and output resolution [s].
    animate        : Show Matplotlib animation after simulation.
    save_animation : File path to save animation (e.g. 'anim.gif'), or None.
    save_plot      : File path to save time-series figure, or None.
    show_plots     : If True, call plt.show() at the end.
    interactive    : If True, launch the interactive swing-up GUI (ignores all
                     other parameters except dt_output and show_plots); returns
                     None in this case because no batch result is produced.

    Returns
    -------
    SimulationResult or None : Time-series data for batch runs, or None when
                               interactive=True (GUI mode produces no batch result).

    How to add a new configuration
    --------------------------------
    1. Write a function config_mytest() that returns a 5-tuple:
           (SystemParameters, MotorModel, PIDController | None,
            initial_state, use_motor_model)
    2. Add it to the CONFIGURATIONS dict below.
    3. Call  main(config_name='mytest')  or pass --config mytest on the CLI.
    """
    if interactive:
        run_interactive(dt=dt_output, show_plots=show_plots)
        return None
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
    params, motor, controller, initial_state, use_motor_model = \
        CONFIGURATIONS[config_name]()

    print(params.summary())
    print()
    print(motor.summary())
    print()
    if controller is not None:
        ctrl_type = type(controller).__name__
        mode = "with motor model" if use_motor_model else "ideal force (motor model bypassed)"
        print(f"{ctrl_type} enabled ({mode}).")
    else:
        print("No controller — free swing.")
    print()

    # ── Build physics engine ──────────────────────────────────────────────────
    # Note: LQR configs build their own physics for gain computation internally.
    # We build a fresh one here for the simulation and visualiser, using the
    # same params — the result is identical.
    physics = TriplePendulumPhysics(params)

    # ── Run simulation ────────────────────────────────────────────────────────
    sim = Simulation(physics, controller, motor)
    result = sim.run(
        t_span=(0.0, t_end),
        initial_state=initial_state,
        dt_output=dt_output,
        use_motor_model=use_motor_model,
    )

    # ── Print summary statistics ──────────────────────────────────────────────
    print("\nResults summary:")
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
    parser.add_argument(
        "--interactive",
        action="store_true",
        help=(
            "Launch the interactive swing-up GUI.  "
            "The pendulum starts hanging; press Start to swing up and balance."
        ),
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
        interactive=args.interactive,
    )
