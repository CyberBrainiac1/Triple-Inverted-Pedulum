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

import sys
import warnings
from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

warnings.filterwarnings("ignore", category=UserWarning)

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent
V2_URDF_PATH = (
    REPO_ROOT
    / "_extracted_urdf_v2"
    / "finaltripleinvertedpendulum"
    / "urdf"
    / "finaltripleinvertedpendulum.urdf"
)

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
    c1: Optional[float] = None   # COM distance from proximal joint [m]
    I1: Optional[float] = None   # COM inertia about the out-of-plane axis [kg·m²]

    # ── Link 2 (middle link) ──────────────────────────────────────────────────
    l2: float = 0.25             # Length [m]
    m2: float = 0.10             # Mass [kg]
    c2: Optional[float] = None
    I2: Optional[float] = None

    # ── Link 3 (top link) ─────────────────────────────────────────────────────
    l3: float = 0.20             # Length [m]
    m3: float = 0.08             # Mass [kg]
    c3: Optional[float] = None
    I3: Optional[float] = None

    # ── Link joint damping ────────────────────────────────────────────────────
    joint_damping: float = 0.001  # Viscous damping at each pivot [N·m·s/rad]

    # ── Environment ───────────────────────────────────────────────────────────
    g: float = 9.81              # Gravitational acceleration [m/s²]

    def __post_init__(self) -> None:
        """
        Auto-compute COM locations and COM inertias for uniform rods.

        Important:
          The rigid-body kinetic energy uses
            T = 1/2 m v_com² + 1/2 I_com ω²
          so `I1/I2/I3` must be moments of inertia about each link's centre of
          mass, not about the pivot. For a uniform rod, I_com = m l² / 12.
        """
        if self.c1 is None:
            self.c1 = self.l1 / 2.0
        if self.c2 is None:
            self.c2 = self.l2 / 2.0
        if self.c3 is None:
            self.c3 = self.l3 / 2.0
        if self.I1 is None:
            self.I1 = self.m1 * self.l1 ** 2 / 12.0
        if self.I2 is None:
            self.I2 = self.m2 * self.l2 ** 2 / 12.0
        if self.I3 is None:
            self.I3 = self.m3 * self.l3 ** 2 / 12.0

    def summary(self) -> str:
        lines = [
            "=== SystemParameters ===",
            f"  Cart: M={self.M_cart:.3f} kg, x∈[{self.x_min},{self.x_max}] m",
            f"  Link1: l={self.l1:.3f} m, c={self.c1:.3f} m, m={self.m1:.3f} kg, I_com={self.I1:.5f} kg·m²",
            f"  Link2: l={self.l2:.3f} m, c={self.c2:.3f} m, m={self.m2:.3f} kg, I_com={self.I2:.5f} kg·m²",
            f"  Link3: l={self.l3:.3f} m, c={self.c3:.3f} m, m={self.m3:.3f} kg, I_com={self.I3:.5f} kg·m²",
            f"  g={self.g} m/s², cart_friction={self.cart_friction} N·s/m",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class V2MechanismDimensions:
    """Physical dimensions extracted from the uploaded V2 URDF asset."""

    link1_m: float
    link2_m: float
    link3_m: float
    rail_limit_m: float
    lead_m_per_rev: float


@dataclass(frozen=True)
class V2MechanismDynamics:
    """
    Effective reduced-order dynamics extracted from the uploaded V2 URDF.

    The actual CAD export contains many fixed hardware components around the
    moving carriage and the first pendulum joint. This reduced model collapses
    each rigidly connected moving cluster into a single body, preserving:
      - exact V2 bar lengths
      - effective moving cart mass
      - link COM locations along each bar
      - link COM inertia about the swing axis
    """

    cart_mass_kg: float
    link1_m: float
    link1_mass_kg: float
    link1_com_m: float
    link1_inertia_kgm2: float
    link2_m: float
    link2_mass_kg: float
    link2_com_m: float
    link2_inertia_kgm2: float
    link3_m: float
    link3_mass_kg: float
    link3_com_m: float
    link3_inertia_kgm2: float
    rail_limit_m: float
    lead_m_per_rev: float


def load_v2_mechanism_dimensions(path: Path = V2_URDF_PATH) -> V2MechanismDimensions:
    """
    Extract the actual linkage lengths from the V2 URDF.

    The exported V2 URDF uses the link inertial origins of `part_5/6/7`
    at roughly the half-length points of the three passive bars, so
    `2 * ||origin_xyz||` yields the full bar length for each link.

    The raw joint limits in the CAD export are placeholder `±10000`.
    The local simulator therefore clamps the physical x travel to `±0.14 m`,
    which matches the real carriage stroke used elsewhere in the repo.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected V2 URDF at '{path}'.")

    root = ET.parse(path).getroot()

    def link_length(link_name: str) -> float:
        link = root.find(f"link[@name='{link_name}']")
        if link is None:
            raise KeyError(f"Link '{link_name}' not found in '{path}'.")
        origin = link.find("inertial/origin")
        if origin is None:
            raise KeyError(f"Link '{link_name}' is missing inertial/origin in '{path}'.")
        xyz = np.array([float(part) for part in origin.attrib["xyz"].split()], dtype=float)
        return float(2.0 * np.linalg.norm(xyz))

    return V2MechanismDimensions(
        link1_m=link_length("part_5"),
        link2_m=link_length("part_6"),
        link3_m=link_length("part_7"),
        rail_limit_m=0.14,
        lead_m_per_rev=8.0e-3,
    )


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


@dataclass
class RotaryMotorModel:
    """
    Brushed DC motor acting directly on a revolute joint.

    The torque-speed law is the same linear DC-motor model used by the
    lead-screw actuator, but the output is a rotary torque instead of a
    linear carriage force.
    """

    free_speed_rpm: float = 312.0
    stall_torque_Nm: float = 5.2

    @classmethod
    def secondary_motor(cls) -> "RotaryMotorModel":
        return cls(free_speed_rpm=312.0, stall_torque_Nm=5.2)

    def torque_from_command(self, command: float, joint_speed_rads: float = 0.0) -> float:
        omega_free = self.free_speed_rpm * 2.0 * np.pi / 60.0
        tau = self.stall_torque_Nm * (command - joint_speed_rads / omega_free)
        return float(np.clip(tau, -self.stall_torque_Nm, self.stall_torque_Nm))


@dataclass
class ActuationCommand:
    """
    Two-input actuation command for the real mechanism abstraction.

    `cart_force` is the desired generalised x-force prior to the lead-screw
    motor model, while `joint1_torque` is the desired torque about the
    driven revolute mate used for the slower secondary motor.
    """

    cart_force: float
    joint1_torque: float = 0.0


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
        self,
        state: np.ndarray,
        F_cart: float,
        tau_joint1: float = 0.0,
        joint1_sign: float = 1.0,
    ) -> np.ndarray:
        """
        Compute state derivatives [q̇, q̈] for the ODE integrator.

        We solve the linear system:
            M · q̈  =  Q − h − G − D

        for the four accelerations q̈ = [ẍ, θ̈1, θ̈2, θ̈3].

        Parameters
        ----------
        state   : [x, θ1, θ2, θ3, ẋ, θ̇1, θ̇2, θ̇3]  (8-element array)
        F_cart     : External force applied to the cart [N] (positive = rightward)
        tau_joint1 : Optional torque applied at the first revolute axis [N·m].
        joint1_sign: Sign mapping from actuator torque to the generalised θ1
                     coordinate. Use `-1.0` for the V2 slower-motor mate.

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

        # External generalised force:
        #   x-axis lead-screw force acts on q[0]
        #   secondary motor torque acts on q[1] with a mechanism-dependent sign
        Q = np.array([F_cart, joint1_sign * tau_joint1, 0.0, 0.0])

        # Solve  M · q̈ = Q − h − G − D  (standard NumPy linear solver)
        rhs = Q - h - G - D
        try:
            q_ddot = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            # Degenerate (near-singular) configuration — use least-squares
            q_ddot, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

        # Hard wall: if the cart hits a travel limit, stop its acceleration
        if (x <= p.x_min and dx < 0) or (x >= p.x_max and dx > 0):
            q_ddot[0] = 0.0

        # Return full state derivative: [q̇, q̈]
        return np.concatenate([state[4:], q_ddot])

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

    def linearise_dual_input(self, joint1_sign: float = 1.0) -> tuple:
        """
        Return the linearised 2-input state-space model for the real mechanism.

        Inputs
        ------
        u[0] = cart generalised force
        u[1] = driven torque about the slower revolute mate

        `joint1_sign` maps actuator torque direction into the θ1 generalised
        coordinate. The V2 URDF path uses `-1.0`.
        """
        M0 = self._mass_matrix(0.0, 0.0, 0.0)
        Minv = np.linalg.inv(M0)

        G_lin = np.diag([
            0.0,
            -self._alpha1 * self.params.g,
            -self._alpha2 * self.params.g,
            -self._alpha3 * self.params.g,
        ])

        A = np.zeros((8, 8))
        A[:4, 4:] = np.eye(4)
        A[4:, :4] = -Minv @ G_lin

        B = np.zeros((8, 2))
        B[4:, 0] = Minv[:, 0]
        B[4:, 1] = joint1_sign * Minv[:, 1]
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
            stability = "stable" if ev.real < 0 else "UNSTABLE"
            print(f"    {ev.real:+.3f} +/- {abs(ev.imag):.3f}j  {stability}")

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


class DualInputLQRController:
    """
    Two-input LQR for the V2 mechanism abstraction.

    Inputs:
      - cart generalised force along the x stage
      - slower secondary-motor torque about the driven revolute mate

    This controller is used for the more physical local simulator path where
    the mechanism has both the x-axis lead-screw drive and the slower
    revolute motor active.
    """

    def __init__(
        self,
        physics: "TriplePendulumPhysics",
        Q_diag: List[float],
        R_diag: List[float],
        max_cart_force: float = 500.0,
        max_joint1_torque: float = 5.2,
        joint1_sign: float = -1.0,
        secondary_motor_model: Optional[RotaryMotorModel] = None,
    ) -> None:
        from scipy.linalg import solve_continuous_are

        if len(Q_diag) != 8:
            raise ValueError("DualInputLQRController expects an 8-element Q_diag.")
        if len(R_diag) != 2:
            raise ValueError("DualInputLQRController expects a 2-element R_diag.")

        self.max_cart_force = max_cart_force
        self.max_joint1_torque = max_joint1_torque
        self.joint1_sign = joint1_sign
        self.secondary_motor_model = secondary_motor_model

        A, B = physics.linearise_dual_input(joint1_sign=joint1_sign)
        Q = np.diag(Q_diag)
        R = np.diag(R_diag)
        P = solve_continuous_are(A, B, Q, R)
        self.K = np.linalg.solve(R, B.T @ P)

        A_cl = A - B @ self.K
        eigs = np.linalg.eigvals(A_cl)
        eigs_sorted = sorted(eigs, key=lambda e: e.real)
        print("  Dual-input LQR closed-loop eigenvalues:")
        for ev in eigs_sorted:
            stability = "stable" if ev.real < 0 else "UNSTABLE"
            print(f"    {ev.real:+.3f} +/- {abs(ev.imag):.3f}j  {stability}")

    def reset(self) -> None:
        pass

    def update_integrals(self, state: np.ndarray, dt: float, x_ref: float = 0.0) -> None:
        pass

    def compute(
        self, state: np.ndarray, dt: float = 0.01, x_ref: float = 0.0
    ) -> ActuationCommand:
        z = state.copy()
        z[0] -= x_ref
        u = -(self.K @ z)
        return ActuationCommand(
            cart_force=float(np.clip(u[0], -self.max_cart_force, self.max_cart_force)),
            joint1_torque=float(np.clip(u[1], -self.max_joint1_torque, self.max_joint1_torque)),
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4c — SwingUpController
# ══════════════════════════════════════════════════════════════════════════════

class SwingUpController:
    """
    Energy-pumping swing-up controller for the triple inverted pendulum.

    Injects mechanical energy into the pendulum chain by moving the cart
    in the direction that maximises the rate of energy increase each half-
    cycle — identical in principle to a parent pushing a child on a swing.

    Energy definition (measured relative to the all-hanging equilibrium)
    --------------------------------------------------------------------
    V = g · [m1·(l1/2)·(1+cos θ1)
           + m2·(l1·(1+cos θ1) + (l2/2)·(1+cos θ2))
           + m3·(l1·(1+cos θ1) + l2·(1+cos θ2) + (l3/2)·(1+cos θ3))]

    T = ½ · (I1·θ̇1² + I2·θ̇2² + I3·θ̇3²)   [rotational KE only]

    E = T + V   → 0 when all links hang still, E_ref when all upright still

    Target energy at the upright equilibrium (θi = 0, θ̇i = 0):
        E_ref = 2·g·(m1·l1/2 + m2·(l1+l2/2) + m3·(l1+l2+l3/2))

    Control law
    -----------
    sig  = θ̇1·cos θ1 + θ̇2·cos θ2 + θ̇3·cos θ3
    F_sw = k_sw · (E − E_ref) · sig

    When E < E_ref (energy deficit):
      • (E − E_ref) < 0
      • sig < 0 when links rotate through bottom (cos θ ≈ −1, θ̇ > 0)
      • Product > 0  → cart force positive (rightward) → injects energy ✓

    Cart centering (prevent wall collision during swing-up):
        F_cen = −k_x·(x − x_ref) − k_dx·ẋ
    """

    def __init__(
        self,
        params: SystemParameters,
        k_sw: float = 40.0,
        k_x: float = 30.0,
        k_dx: float = 5.0,
        max_force: float = 600.0,
    ) -> None:
        self.params = params
        self.k_sw = k_sw
        self.k_x = k_x
        self.k_dx = k_dx
        self.max_force = max_force
        p = params

        # Target energy: all links upright with zero angular velocities
        self.E_ref = 2.0 * p.g * (
            p.m1 * (p.l1 / 2.0)
            + p.m2 * (p.l1 + p.l2 / 2.0)
            + p.m3 * (p.l1 + p.l2 + p.l3 / 2.0)
        )
        print(f"  SwingUp: E_ref = {self.E_ref:.4f} J  (energy needed to reach upright)")

    def reset(self) -> None:
        pass  # no internal state

    def update_integrals(
        self, state: np.ndarray, dt: float, x_ref: float = 0.0
    ) -> None:
        pass  # no integral terms

    def _energy(self, state: np.ndarray) -> float:
        """Total mechanical energy relative to the all-hanging position [J]."""
        p = self.params
        _, th1, th2, th3, _, dth1, dth2, dth3 = state
        g = p.g

        # Rotational kinetic energy of the three links
        T = 0.5 * (p.I1 * dth1 ** 2 + p.I2 * dth2 ** 2 + p.I3 * dth3 ** 2)

        # Potential energy above the all-hanging reference level
        # (1 + cos θ) = 0 at hanging (θ=π), 2 at upright (θ=0)
        V = g * (
            p.m1 * (p.l1 / 2.0) * (1.0 + np.cos(th1))
            + p.m2 * (
                p.l1 * (1.0 + np.cos(th1))
                + (p.l2 / 2.0) * (1.0 + np.cos(th2))
            )
            + p.m3 * (
                p.l1 * (1.0 + np.cos(th1))
                + p.l2 * (1.0 + np.cos(th2))
                + (p.l3 / 2.0) * (1.0 + np.cos(th3))
            )
        )
        return T + V

    def compute(
        self, state: np.ndarray, dt: float = 0.01, x_ref: float = 0.0
    ) -> float:
        x, th1, th2, th3, dx, dth1, dth2, dth3 = state

        # Signed energy deficit (negative → below target → need to inject energy)
        dE = self._energy(state) - self.E_ref

        # Energy injection signal: direction that maximises power into pendulum.
        # Each term is saturated at ±max_spin so a rapidly-spinning link (which
        # already has more than its share of energy) doesn't dominate the signal
        # and push the cart far off-centre.
        # Saturate angular-velocity contribution so rapidly-spinning links don't
        # produce an enormous signal that over-pumps the system.
        max_spin = 6.0
        sig = (float(np.clip(dth1, -max_spin, max_spin)) * np.cos(th1)
               + float(np.clip(dth2, -max_spin, max_spin)) * np.cos(th2)
               + float(np.clip(dth3, -max_spin, max_spin)) * np.cos(th3))

        # Self-regulating energy control:
        #   dE < 0 → below target → pump (add energy)
        #   dE > 0 → above target → brake (remove energy)
        #   |dE| < 3% threshold → dead zone (let the system coast)
        #
        # The clamp is SYMMETRIC so braking is exactly as strong as pumping.
        # This prevents the asymmetric over-pumping that causes runaway spin.
        if abs(dE) < 0.03 * self.E_ref:
            F_sw = 0.0
        else:
            dE_clamped = float(np.clip(dE, -self.E_ref, self.E_ref))
            F_sw = self.k_sw * dE_clamped * sig

        # Cart centering: grows super-linearly so the cart can't escape the track
        p = self.params
        half_track = (p.x_max - p.x_min) / 2.0
        x_norm = (x - x_ref) / half_track        # ±1 at the rails
        F_cen = -(self.k_x * x_norm ** 3 * half_track
                  + self.k_x * 0.4 * (x - x_ref)) \
                - self.k_dx * dx

        return float(np.clip(F_sw + F_cen, -self.max_force, self.max_force))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4d — HybridController  (swing-up → LQR catch)
# ══════════════════════════════════════════════════════════════════════════════

class HybridController:
    """
    Two-phase hybrid controller: energy-pumping swing-up → LQR balance.

    Phase 1 — SwingUpController
    ---------------------------
    Pumps mechanical energy into the pendulum chain.  The pendulum oscillates
    with growing amplitude until it passes close to the upright position.

    Phase 2 — LQRController  (triggered on capture)
    -----------------------------------------------
    Once all three link angles are within `capture_deg` of the upright AND
    all angular velocities are below `capture_vel_rads`, control transfers
    to the pre-computed LQR.  The LQR then holds the triple pendulum balanced
    indefinitely.

    The mode is one-way: swing_up → lqr.  Once captured, no reversion.

    Adaptive capture threshold
    --------------------------
    If no capture occurs within `t_relax` seconds, the angle threshold is
    widened by 50 % (capped at `max_capture_deg`) — useful for parameter
    sets where the natural chaotic motion only briefly grazes the upright.
    """

    def __init__(
        self,
        swing_up: SwingUpController,
        lqr: LQRController,
        capture_deg: float = 20.0,
        capture_vel_rads: float = 6.0,
        t_relax: float = 20.0,
        max_capture_deg: float = 30.0,
    ) -> None:
        self.swing_up = swing_up
        self.lqr = lqr
        self.capture_deg = capture_deg
        self.capture_vel_rads = capture_vel_rads
        self.t_relax = t_relax
        self.max_capture_deg = max_capture_deg

        self._mode: str = "swing_up"
        self._switch_time: Optional[float] = None
        self._t_elapsed: float = 0.0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def switch_time(self) -> Optional[float]:
        """Time [s] at which swing-up gave way to LQR, or None if not yet."""
        return self._switch_time

    def reset(self) -> None:
        self._mode = "swing_up"
        self._switch_time = None
        self._t_elapsed = 0.0
        self.swing_up.reset()
        self.lqr.reset()

    @staticmethod
    def _wrap(th: float) -> float:
        """Wrap angle to (−π, π]."""
        return th - 2.0 * np.pi * round(th / (2.0 * np.pi))

    def update_integrals(
        self, state: np.ndarray, dt: float, x_ref: float = 0.0
    ) -> None:
        self._t_elapsed += dt

        if self._mode == "swing_up":
            # Adaptive threshold: loosen after t_relax seconds without capture
            threshold = self.capture_deg
            if self._t_elapsed > self.t_relax:
                threshold = min(self.capture_deg * 1.5, self.max_capture_deg)

            angles  = [self._wrap(float(th)) for th in state[1:4]]
            ang_vel = state[5:8]

            all_near = all(abs(a) < np.radians(threshold) for a in angles)
            all_slow = all(abs(w) < self.capture_vel_rads for w in ang_vel)

            if all_near and all_slow:
                self._mode = "lqr"
                self._switch_time = self._t_elapsed
                print(
                    f"\n  *** CAPTURED at t = {self._t_elapsed:.3f} s -> LQR engaged ***"
                )
                print(
                    f"      angles  (°): "
                    f"{[f'{np.degrees(a):+.1f}' for a in angles]}"
                )
                print(
                    f"      ang vel (rad/s): "
                    f"{[f'{float(w):+.2f}' for w in ang_vel]}"
                )
                self.lqr.reset()
        else:
            self.lqr.update_integrals(state, dt, x_ref)

    def compute(
        self, state: np.ndarray, dt: float = 0.01, x_ref: float = 0.0
    ) -> float:
        if self._mode == "swing_up":
            return self.swing_up.compute(state, dt, x_ref)
        return self.lqr.compute(state, dt, x_ref)


@dataclass
class PotentiometerObserverConfig:
    """
    Configuration for the two potentiometer channels on joints 2 and 3.

    Assumptions
    -----------
    - Joint 1 and the cart states are otherwise observable in simulation.
    - Joint 2 and joint 3 angles are measured with rotary potentiometers.
    - Joint 2 and joint 3 angular velocities are estimated numerically from
      successive angle samples and low-pass filtered.
    """

    sample_hz: float = 250.0
    adc_bits: int = 12
    noise_std_deg: float = 0.20
    theta2_bias_deg: float = 0.0
    theta3_bias_deg: float = 0.0
    velocity_alpha: float = 0.35
    use_true_angular_rates: bool = True
    seed: int = 7


class TwoPotentiometerObserver:
    """
    Sensor model for the upper two linkage joints.

    The observer returns an 8-element state vector compatible with the
    existing controllers:
      [x, θ1, θ2_meas, θ3_meas, ẋ, θ̇1, θ̇2_est, θ̇3_est]
    """

    def __init__(self, config: Optional[PotentiometerObserverConfig] = None) -> None:
        self.config = config or PotentiometerObserverConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._sample_period = 1.0 / max(self.config.sample_hz, 1e-6)
        self.reset()

    @staticmethod
    def _wrap(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _quantize_angle(self, angle: float) -> float:
        levels = float(2 ** max(self.config.adc_bits, 1))
        step = 2.0 * np.pi / levels
        return self._wrap(step * np.round(angle / step))

    def _measure_angle(self, true_angle: float, bias_deg: float) -> float:
        noise = self._rng.normal(0.0, np.radians(self.config.noise_std_deg))
        biased = true_angle + np.radians(bias_deg) + noise
        return self._quantize_angle(biased)

    def reset(self) -> None:
        self._time_since_sample = 0.0
        self._last_theta2: Optional[float] = None
        self._last_theta3: Optional[float] = None
        self._dtheta2_est = 0.0
        self._dtheta3_est = 0.0
        self.last_observation: Optional[np.ndarray] = None

    def observe(self, true_state: np.ndarray, dt: float) -> np.ndarray:
        self._time_since_sample += dt

        if self.last_observation is None:
            self._time_since_sample = self._sample_period

        if self._time_since_sample >= self._sample_period:
            self._time_since_sample = 0.0
            theta2 = self._measure_angle(true_state[2], self.config.theta2_bias_deg)
            theta3 = self._measure_angle(true_state[3], self.config.theta3_bias_deg)

            if self._last_theta2 is not None:
                dtheta2_raw = self._wrap(theta2 - self._last_theta2) / self._sample_period
                dtheta3_raw = self._wrap(theta3 - self._last_theta3) / self._sample_period
                alpha = float(np.clip(self.config.velocity_alpha, 0.0, 1.0))
                self._dtheta2_est = (1.0 - alpha) * self._dtheta2_est + alpha * dtheta2_raw
                self._dtheta3_est = (1.0 - alpha) * self._dtheta3_est + alpha * dtheta3_raw

            self._last_theta2 = theta2
            self._last_theta3 = theta3

            self.last_observation = np.array(
                [
                    true_state[0],
                    true_state[1],
                    theta2,
                    theta3,
                    true_state[4],
                    true_state[5],
                    true_state[6] if self.config.use_true_angular_rates else self._dtheta2_est,
                    true_state[7] if self.config.use_true_angular_rates else self._dtheta3_est,
                ],
                dtype=float,
            )

        return self.last_observation.copy()


class MeasuredStateController:
    """
    Wraps an existing controller so it acts on observed rather than true state.

    This is the simplest way to test hardware-oriented sensing assumptions in
    the physics simulator without rewriting the controllers themselves.
    """

    def __init__(self, base_controller, observer: TwoPotentiometerObserver) -> None:
        self.base_controller = base_controller
        self.observer = observer
        self.last_observation: Optional[np.ndarray] = None

    @property
    def mode(self) -> Optional[str]:
        return getattr(self.base_controller, "mode", None)

    @property
    def switch_time(self) -> Optional[float]:
        return getattr(self.base_controller, "switch_time", None)

    def reset(self) -> None:
        self.observer.reset()
        self.last_observation = None
        self.base_controller.reset()

    def update_integrals(
        self, state: np.ndarray, dt: float, x_ref: float = 0.0
    ) -> None:
        measured = self.observer.observe(state, dt)
        self.last_observation = measured.copy()
        self.base_controller.update_integrals(measured, dt, x_ref)

    def compute(
        self, state: np.ndarray, dt: float = 0.01, x_ref: float = 0.0
    ) -> float:
        measured = self.observer.observe(state, dt)
        self.last_observation = measured.copy()
        return self.base_controller.compute(measured, dt, x_ref)


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
    observed_states: Optional[np.ndarray] = None   # Controller-visible state history
    secondary_torque: Optional[np.ndarray] = None   # Applied slower-motor torque [N·m]


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
        secondary_motor = getattr(self.controller, "secondary_motor_model", None)
        joint1_sign = getattr(self.controller, "joint1_sign", 1.0)
        stall_tau = secondary_motor.stall_torque_Nm if secondary_motor else 1.0

        t_start, t_end = t_span
        t_steps = np.arange(t_start, t_end, dt_output)

        # Pre-allocate output arrays
        n = len(t_steps)
        out_states = np.empty((n, 8))
        out_observed_states = np.empty((n, 8))
        out_forces = np.empty(n)
        out_secondary_torques = np.zeros(n)

        state = np.array(initial_state, dtype=float)
        print(f"Integrating t=[{t_start}, {t_end}] s  ({n} control steps) ...")

        for k, t_now in enumerate(t_steps):
            t_next = min(t_now + dt_output, t_end)

            # ── 1. Evaluate control force at current state ─────────────────
            if self.controller is not None:
                raw_command = self.controller.compute(state, dt_output)
                if isinstance(raw_command, ActuationCommand):
                    f_cmd = raw_command.cart_force
                    tau_cmd = raw_command.joint1_torque
                else:
                    f_cmd = float(raw_command)
                    tau_cmd = 0.0
            else:
                f_cmd = 0.0
                tau_cmd = 0.0

            # ── 2. Pass through motor model(s) (back-EMF, saturation) ──────
            if use_motor_model and self.motor is not None:
                # Normalise desired force to a voltage command [−1, +1]
                command = float(np.clip(f_cmd / stall_f, -1.0, 1.0))
                F_cart  = self.motor.force_from_command(command, cart_speed_ms=state[4])
            else:
                F_cart = f_cmd

            if use_motor_model and secondary_motor is not None:
                command_tau = float(np.clip(tau_cmd / stall_tau, -1.0, 1.0))
                tau_joint1 = secondary_motor.torque_from_command(
                    command_tau,
                    joint_speed_rads=state[5],
                )
            else:
                tau_joint1 = tau_cmd

            # Record output for this step
            out_states[k] = state
            observed = getattr(self.controller, "last_observation", None)
            out_observed_states[k] = (
                state if observed is None else np.array(observed, dtype=float)
            )
            out_forces[k] = F_cart
            out_secondary_torques[k] = tau_joint1

            # ── 3. Integrate ODE for one control interval (F is constant) ──
            # The ODE function is now stateless within this interval.
            def ode_rhs(
                t: float,
                s: np.ndarray,
                _F: float = F_cart,
                _tau: float = tau_joint1,
            ) -> np.ndarray:
                return self.physics.equations_of_motion(
                    s,
                    _F,
                    tau_joint1=_tau,
                    joint1_sign=joint1_sign,
                )

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

            # ── 3b. Hard cart wall enforcement (fully inelastic) ──────────
            # The EOM only zeros acceleration at the wall; the cart still drifts
            # past at its current velocity.  Clamp position and zero outward
            # velocity so the cart can't escape the track.
            p = self.physics.params
            if state[0] < p.x_min:
                state[0] = p.x_min
                state[4] = max(0.0, state[4])   # kill leftward vel, keep rightward
            elif state[0] > p.x_max:
                state[0] = p.x_max
                state[4] = min(0.0, state[4])   # kill rightward vel, keep leftward

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
            observed_states=out_observed_states,
            secondary_torque=out_secondary_torques,
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

    def __init__(
        self,
        physics: TriplePendulumPhysics,
        result: SimulationResult,
        switch_time: Optional[float] = None,
    ) -> None:
        self.physics = physics
        self.result = result
        self._p = physics.params
        self.switch_time = switch_time   # LQR catch time for HybridController runs

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

        # Y-axis spans the full pendulum range (hanging down through upright)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(p.x_min - margin, p.x_max + margin)
        ax.set_ylim(-total_len - margin, total_len + margin)
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
            0.02, 0.97, "", transform=ax.transAxes, fontsize=10, va="top"
        )

        # Controller mode label (shows "Swing-up" or "LQR Balance")
        mode_text = ax.text(
            0.02, 0.91, "", transform=ax.transAxes, fontsize=10, va="top",
            fontweight="bold",
        )

        # Angle readouts
        angle_text = ax.text(
            0.72, 0.97, "", transform=ax.transAxes, fontsize=9, va="top",
            family="monospace"
        )

        # Subsample for animation frame rate
        n_frames = len(r.t)
        step = max(1, int(n_frames / (r.t[-1] * 1000 / interval_ms / speed_factor)))
        sw_t = self.switch_time  # capture once to avoid closure issues

        def init():
            pivot_dot.set_data([], [])
            for ln in link_lines:
                ln.set_data([], [])
            time_text.set_text("")
            mode_text.set_text("")
            angle_text.set_text("")
            return link_lines + [cart_patch, pivot_dot, time_text, mode_text, angle_text]

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

            t_now = r.t[i]
            time_text.set_text(f"t = {t_now:.3f} s")

            # Update controller mode label
            if sw_t is None or t_now < sw_t:
                mode_text.set_text("Phase: Swing-up (energy pumping)")
                mode_text.set_color("darkorange")
            else:
                mode_text.set_text("Phase: LQR Balance")
                mode_text.set_color("mediumseagreen")

            angle_text.set_text(
                f"θ1={np.degrees(state[1]):+6.1f}°\n"
                f"θ2={np.degrees(state[2]):+6.1f}°\n"
                f"θ3={np.degrees(state[3]):+6.1f}°"
            )
            return link_lines + [cart_patch, pivot_dot, time_text, mode_text, angle_text]

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
            print(f"Saving animation to {save_path} ...")
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
        sw = self.switch_time  # None for non-hybrid runs
        observed = r.observed_states

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        title = "Triple Inverted Pendulum — Time Series"
        if sw is not None:
            title += f"  (swing-up → LQR at t = {sw:.2f} s)"
        fig.suptitle(title, fontsize=13)

        def _vline(ax: plt.Axes) -> None:
            """Draw phase-transition marker on an axis."""
            if sw is not None:
                ax.axvline(
                    sw, color="royalblue", linewidth=1.5,
                    linestyle="--", alpha=0.8, label=f"LQR catch (t={sw:.1f} s)"
                )

        # ── Panel 1: angles ───────────────────────────────────────────────
        ax = axes[0]
        ax.plot(t, np.degrees(r.theta1), label="θ1 (bottom)", color="tomato")
        ax.plot(t, np.degrees(r.theta2), label="θ2 (middle)", color="goldenrod")
        ax.plot(t, np.degrees(r.theta3), label="θ3 (top)",    color="mediumseagreen")
        if observed is not None:
            ax.plot(
                t, np.degrees(observed[:, 2]),
                label="θ2 meas", color="goldenrod", linestyle=":", alpha=0.8,
            )
            ax.plot(
                t, np.degrees(observed[:, 3]),
                label="θ3 meas", color="mediumseagreen", linestyle=":", alpha=0.8,
            )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        _vline(ax)
        ax.set_ylabel("Angle [°]")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Panel 2: angular velocities ───────────────────────────────────
        ax = axes[1]
        ax.plot(t, np.degrees(r.dtheta1), label="θ̇1", color="tomato",      linestyle="--")
        ax.plot(t, np.degrees(r.dtheta2), label="θ̇2", color="goldenrod",   linestyle="--")
        ax.plot(t, np.degrees(r.dtheta3), label="θ̇3", color="mediumseagreen", linestyle="--")
        if observed is not None:
            ax.plot(
                t, np.degrees(observed[:, 6]),
                label="θ̇2 est", color="goldenrod", linestyle=":", alpha=0.8,
            )
            ax.plot(
                t, np.degrees(observed[:, 7]),
                label="θ̇3 est", color="mediumseagreen", linestyle=":", alpha=0.8,
            )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        _vline(ax)
        ax.set_ylabel("Angular vel. [°/s]")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Panel 3: cart position and velocity ───────────────────────────
        ax = axes[2]
        ax.plot(t, r.x,  label="x (position)", color="steelblue")
        ax.plot(t, r.dx, label="ẋ (velocity)",  color="steelblue", linestyle="--", alpha=0.7)
        ax.axhline(0,             color="gray",   linewidth=0.8, linestyle="--")
        ax.axhline(self._p.x_min, color="salmon", linewidth=0.8, linestyle=":")
        ax.axhline(self._p.x_max, color="salmon", linewidth=0.8, linestyle=":")
        _vline(ax)
        ax.set_ylabel("Cart x [m] / ẋ [m/s]")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

        # ── Panel 4: control force ────────────────────────────────────────
        ax = axes[3]
        ax.plot(t, r.force, label="F_cart", color="purple")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        _vline(ax)
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

    print("Computing LQR gains (pd_stabilise)...")
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

    print("Computing LQR gains (longer_links)...")
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

    print("Computing LQR gains (secondary_motor)...")
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


def config_hardware_balance() -> tuple:
    """
    Configuration E — Hardware-length balance demo with a larger local rail.

    This is the most reliable local configuration for controller and sensor
    development.  It uses the requested link lengths and a local-only rail
    range large enough for the aggressive balance controller to recover.
    """
    params = SystemParameters(
        M_cart=2.0,
        l1=0.20, m1=0.15,
        l2=0.15, m2=0.10,
        l3=0.10, m3=0.08,
        x_min=-3.0, x_max=3.0,
        cart_friction=1.0,
        joint_damping=0.0005,
    )
    motor = MotorModel()
    physics = TriplePendulumPhysics(params)

    print("Computing LQR gains for hardware_balance ...")
    controller = LQRController(
        physics=physics,
        Q_diag=[10.0, 8000.0, 8000.0, 8000.0, 1.0, 400.0, 400.0, 400.0],
        R=0.0001,
        max_force=600.0,
    )

    initial_state = np.array([
        0.0,
        np.radians(8.0),
        np.radians(8.0),
        np.radians(8.0),
        0.0, 0.0, 0.0, 0.0,
    ])
    return params, motor, controller, initial_state, False


def config_v2_dual_balance() -> tuple:
    """
    Configuration F — Exact V2-URDF geometry with dual real-actuator physics.

    Uses:
      - x-axis actuation through the V2 `x_axis_1 / x_axis_2` lead-screw path
      - slower rotary actuation through the V2 secondary revolute mate
      - passive link lengths extracted directly from the uploaded V2 URDF

    This is the most physically grounded local balance mode currently in the
    repo. The raw CAD-export travel limits are placeholders, so the simulator
    enforces the physical rail range of ±0.14 m.
    """
    dims = load_v2_mechanism_dimensions()
    params = SystemParameters(
        M_cart=2.0,
        l1=dims.link1_m, m1=0.15,
        l2=dims.link2_m, m2=0.10,
        l3=dims.link3_m, m3=0.08,
        x_min=-dims.rail_limit_m, x_max=dims.rail_limit_m,
        cart_friction=1.0,
        joint_damping=0.005,
    )
    motor = MotorModel(
        lead_mm=dims.lead_m_per_rev * 1e3,
        max_force_N=700.0,
    )
    physics = TriplePendulumPhysics(params)

    print("Computing dual-input LQR gains for v2_dual_balance ...")
    controller = DualInputLQRController(
        physics=physics,
        Q_diag=[
            3.8344152780846716,
            334.4171731531393,
            773.2118516660596,
            28202.6905434322,
            1.4150798590751337,
            19.488663673184387,
            10.241948453505868,
            79.40920931463543,
        ],
        R_diag=[0.09981155354073941, 0.0021478724678999604],
        max_cart_force=4000.0,
        max_joint1_torque=5.2,
        joint1_sign=-1.0,
        secondary_motor_model=RotaryMotorModel.secondary_motor(),
    )

    # Verified local basin for the dual-input, motor-limited V2 model.
    initial_state = np.array([
        0.0,
        np.radians(1.0),
        np.radians(1.0),
        np.radians(1.0),
        0.0, 0.0, 0.0, 0.0,
    ])
    return params, motor, controller, initial_state, True


def config_swing_up() -> tuple:
    """
    Configuration F — Full swing-up from rest → LQR balance (hardware dimensions).

    Hardware link dimensions: l1=0.20 m, l2=0.15 m, l3=0.10 m.

    Starts with all three links hanging straight down (θ1=θ2=θ3=π rad) and
    a small asymmetric angular velocity kick.  An energy-pumping
    SwingUpController oscillates the cart to inject mechanical energy into the
    pendulum chain.  Once all three link angles are simultaneously within 20°
    of the upright AND angular velocities are below 6 rad/s, the
    HybridController switches to LQR to hold balance.

    End-to-end pipeline
    -------------------
    rest (hanging) → energy-pumping swing-up → LQR catch & balance

    Recommended runtime:  30 s
        python triple_pendulum_simulation.py --config swing_up --t_end 30

    Physics notes
    -------------
    • Cart travel extended to ±3.0 m for the local-only swing-up sandbox.
    • Joint damping 0.0005 N·m·s/rad — low so swings build amplitude quickly.
    • Motor model bypassed (ideal force) for reliable LQR catch.
    • E_ref ≈ 1.46 J  (energy to raise all three links from hanging to upright).
    """
    params = SystemParameters(
        M_cart=2.0,
        l1=0.20, m1=0.15,   # hardware link 1
        l2=0.15, m2=0.10,   # hardware link 2
        l3=0.10, m3=0.08,   # hardware link 3
        x_min=-3.0, x_max=3.0,
        cart_friction=1.0,
        joint_damping=0.0005,
    )
    motor = MotorModel()
    physics = TriplePendulumPhysics(params)

    print("Computing LQR gains for balance phase ...")
    lqr = LQRController(
        physics=physics,
        Q_diag=[10.0, 8000.0, 8000.0, 8000.0, 1.0, 400.0, 400.0, 400.0],
        R=0.0001,
        max_force=600.0,
    )

    swing_up = SwingUpController(
        params=params,
        k_sw=12.0,    # gentle energy-pumping gain — prevents over-spin
        k_x=60.0,     # centering stiffness
        k_dx=10.0,    # centering damping
        max_force=600.0,
    )

    controller = HybridController(
        swing_up=swing_up,
        lqr=lqr,
        capture_deg=18.0,        # angle window for LQR catch
        capture_vel_rads=4.5,    # max angular speed at catch (LQR can handle this)
        t_relax=30.0,
        max_capture_deg=25.0,
    )

    # All links start on the SAME side (θ slightly below π) so they swing in
    # phase — greatly increases the chance of a simultaneous near-upright event.
    # Zero velocities = true "start from rest".
    initial_state = np.array([
        0.0,                         # cart at centre
        np.pi - np.radians(8),       # θ1: 8° off toward upright
        np.pi - np.radians(8),       # θ2: same side
        np.pi - np.radians(8),       # θ3: same side
        0.0, 0.0, 0.0, 0.0,         # all velocities zero (true rest)
    ])

    return params, motor, controller, initial_state, False


# Map of named configurations for easy selection on the command line
CONFIGURATIONS = {
    "free_swing":       config_free_swing,
    "pd_stabilise":     config_pd_stabilise,
    "longer_links":     config_longer_links,
    "secondary_motor":  config_secondary_motor,
    "hardware_balance": config_hardware_balance,
    "v2_dual_balance":  config_v2_dual_balance,
    "swing_up":         config_swing_up,
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — main()
# ══════════════════════════════════════════════════════════════════════════════

def main(
    config_name: str = "hardware_balance",
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
    dt_output      : Control step size and output resolution [s].
    animate        : Show Matplotlib animation after simulation.
    save_animation : File path to save animation (e.g. 'anim.gif'), or None.
    save_plot      : File path to save time-series figure, or None.
    show_plots     : If True, call plt.show() at the end.

    Returns
    -------
    SimulationResult : All time-series data.

    How to add a new configuration
    --------------------------------
    1. Write a function config_mytest() that returns a 5-tuple:
           (SystemParameters, MotorModel, PIDController | None,
            initial_state, use_motor_model)
    2. Add it to the CONFIGURATIONS dict below.
    3. Call  main(config_name='mytest')  or pass --config mytest on the CLI.
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
        print("No controller - free swing.")
    print()

    # ── Build physics engine ──────────────────────────────────────────────────
    # Note: LQR configs build their own physics for gain computation internally.
    # We build a fresh one here for the simulation and visualiser, using the
    # same params — the result is identical.
    physics = TriplePendulumPhysics(params)

    # ── Run simulation ────────────────────────────────────────────────────────
    # The short-link local swing-up case is numerically stiff enough that it
    # benefits from a faster controller / output step than the other configs.
    if config_name == "swing_up" and dt_output > 0.002:
        print(
            "  [Note] swing_up config is running with a coarse dt. "
            "Use --dt 0.002 for the best local balance-phase behaviour.\n"
        )

    # For swing_up the default t_end of 5 s is too short; suggest 30 s.
    if config_name == "swing_up" and t_end < 20.0:
        print(
            "  [Note] swing_up config: t_end is short.  "
            "Consider --t_end 60 for a full swing-up + balance run.\n"
        )

    sim = Simulation(physics, controller, motor)
    result = sim.run(
        t_span=(0.0, t_end),
        initial_state=initial_state,
        dt_output=dt_output,
        use_motor_model=use_motor_model,
    )

    # ── Extract switch time from HybridController (if applicable) ────────────
    switch_time: Optional[float] = None
    if isinstance(controller, HybridController):
        switch_time = controller.switch_time
        if switch_time is None:
            print(
                "\n  [Warning] HybridController never captured — "
                "LQR was not engaged.  Try a longer run (--t_end 30)."
            )

    # ── Print summary statistics ──────────────────────────────────────────────
    print("\nResults summary:")
    if switch_time is not None:
        print(f"  Swing-up → LQR capture at  t = {switch_time:.3f} s")
    print(f"  Cart x range: [{result.x.min():.4f}, {result.x.max():.4f}] m")
    print(f"  θ1 range: [{np.degrees(result.theta1.min()):.1f}°, "
          f"{np.degrees(result.theta1.max()):.1f}°]")
    print(f"  θ2 range: [{np.degrees(result.theta2.min()):.1f}°, "
          f"{np.degrees(result.theta2.max()):.1f}°]")
    print(f"  θ3 range: [{np.degrees(result.theta3.min()):.1f}°, "
          f"{np.degrees(result.theta3.max()):.1f}°]")
    print(f"  Max |F_cart|: {np.max(np.abs(result.force)):.1f} N")

    # ── Visualise ─────────────────────────────────────────────────────────────
    viz = Visualizer(physics, result, switch_time=switch_time)

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
        default="hardware_balance",
        choices=list(CONFIGURATIONS.keys()),
        help=(
            "Named test configuration to run (default: hardware_balance). "
            "Choices: " + ", ".join(CONFIGURATIONS.keys())
        ),
    )
    parser.add_argument(
        "--t_end", type=float, default=None,
        help=(
            "Simulation duration in seconds. "
            "Defaults to 30 s for swing_up, 5 s for all other configs."
        ),
    )
    parser.add_argument(
        "--dt", type=float, default=None,
        help=(
            "Output time resolution in seconds. "
            "Defaults to 0.002 s for hardware_balance and swing_up, "
            "and 0.01 s for other configs."
        ),
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

    # Auto-set t_end: swing_up needs more time than the other configs
    t_end = args.t_end
    if t_end is None:
        t_end = 60.0 if args.config == "swing_up" else 5.0

    dt_output = args.dt
    if dt_output is None:
        dt_output = 0.002 if args.config in {"hardware_balance", "swing_up"} else 0.01

    main(
        config_name=args.config,
        t_end=t_end,
        dt_output=dt_output,
        animate=not args.no_animate,
        save_animation=args.save_anim,
        save_plot=args.save_plot,
        show_plots=not args.no_show,
    )
