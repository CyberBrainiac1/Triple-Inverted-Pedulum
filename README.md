# Triple Inverted Pendulum on a Cart — Physics Simulation

A self-contained Python simulation of a **triple inverted pendulum on a motorised cart**, complete with Lagrangian physics, an LQR stabilising controller, a brushed DC lead-screw motor model, and a Matplotlib animation.

---

## Quick start

```bash
pip install numpy scipy matplotlib "mujoco<3.2"

# Local UI with the hardware-length physics model and potentiometer channels
python triple_pendulum_ui.py

# Watch the LQR controller stabilise the pendulum
python triple_pendulum_simulation.py --config hardware_balance --t_end 5

# Free fall (no control) — watch the physics
python triple_pendulum_simulation.py --config free_swing --t_end 3

# All options
python triple_pendulum_simulation.py --help
```

---

## Table of contents

1. [What this simulation models](#what-this-simulation-models)
2. [Physics concepts explained simply](#physics-concepts-explained-simply)
   - [Generalised coordinates](#generalised-coordinates)
   - [The Lagrangian — kinetic and potential energy](#the-lagrangian)
   - [Mass matrix M(q)](#mass-matrix-mq)
   - [Coriolis and centrifugal terms h(q, q̇)](#coriolis-and-centrifugal-terms)
   - [Gravity vector G(q)](#gravity-vector-gq)
   - [Putting it together — the equation of motion](#putting-it-together)
   - [Damping](#damping)
3. [Motor model — brushed DC lead screw](#motor-model)
4. [Control — PID and LQR](#control)
   - [PID controller (manual tuning)](#pid-controller)
   - [LQR controller (automatic optimal gains)](#lqr-controller)
5. [Code structure](#code-structure)
6. [How to change parameters](#how-to-change-parameters)
7. [Configurations](#configurations)
8. [Output and visualisation](#output-and-visualisation)
9. [Frequently asked questions](#faq)

---

## What this simulation models

```
          ●  ← tip of link 3
          |  link 3 (length l3, mass m3)
          ●  ← pivot between link 2 and 3
          |  link 2 (length l2, mass m2)
          ●  ← pivot between link 1 and 2
          |  link 1 (length l1, mass m1)
    ┌─────┼─────┐
    │   CART    │ ← mass M_cart
    └─────┼─────┘
          |
    ══════╪════════════════  ← rail / track
          x →
```

- The **cart** slides left and right along a horizontal rail.
- **Three rigid links** are connected end-to-end above the cart, each free to rotate at its lower pivot.
- A **brushed DC lead-screw motor** pushes the cart; the controller decides how hard and in which direction.
- All three links start near the upright (balanced) position. Without control they immediately fall over — the system is naturally unstable. With an LQR controller the pendulum is held upright.

---

## Physics concepts explained simply

### Generalised coordinates

Instead of tracking X and Y positions of every point, we describe the whole system with just **four numbers**:

| Symbol | What it means |
|--------|---------------|
| `x`    | Cart position along the rail (metres, positive = right) |
| `θ₁`   | Angle of link 1 from straight-up (radians, positive = leaning right) |
| `θ₂`   | Angle of link 2 from straight-up |
| `θ₃`   | Angle of link 3 from straight-up |

These four numbers — called **generalised coordinates** and written `q = [x, θ₁, θ₂, θ₃]` — completely describe the position of every part of the system. Their time-derivatives `q̇ = [ẋ, θ̇₁, θ̇₂, θ̇₃]` describe how fast everything is moving.

### The Lagrangian

The **Lagrangian method** is an elegant way to derive equations of motion from energy. Instead of drawing free-body diagrams and tracking every force, we write:

```
Lagrangian  L = T − V
```

where:
- **T** = total kinetic energy of the system (cart + all three links)
- **V** = total potential energy (gravitational height of all three links)

The equations of motion then follow automatically from the formula:

```
d/dt (∂L/∂q̇ᵢ) − ∂L/∂qᵢ = Qᵢ   for each coordinate i
```

where `Qᵢ` is the external force/torque acting on coordinate `i`.  For our system `Q = [F_cart, 0, 0, 0]` — only the cart has an external force (from the motor); the pivot joints are frictionless.

#### Kinetic energy

Each link has **translational** and **rotational** kinetic energy:

```
T_link_i = ½ mᵢ (ẋ_cm² + ẏ_cm²) + ½ Iᵢ θ̇ᵢ²
```

- `(ẋ_cm, ẏ_cm)` = velocity of the link's centre of mass  
- `Iᵢ = mᵢ lᵢ² / 3` = moment of inertia of a uniform rod about one end

The centre of mass of link `i` moves because:
1. The cart moves (the whole pendulum base moves with x)
2. All the links below it rotate (changing the pivot position of link `i`)
3. Link `i` itself rotates

This coupling is why the mass matrix has off-diagonal terms.

#### Potential energy

```
V = m₁ g (l₁/2) cos θ₁
  + m₂ g (l₁ cos θ₁ + (l₂/2) cos θ₂)
  + m₃ g (l₁ cos θ₁ + l₂ cos θ₂ + (l₃/2) cos θ₃)
```

Height increases going upward, so `V` is maximum when all links are upright (`θ = 0`). This is the unstable equilibrium — any small tilt makes gravity pull the link further away.

### Mass matrix M(q)

After working through the Lagrangian algebra (see the code for step-by-step derivation), the equations of motion take the compact matrix form:

```
M(q) · q̈  =  Q − h(q, q̇) − G(q) − D
```

`M(q)` is the **4 × 4 symmetric mass matrix**. It captures how forces and torques in the system are "spread around" between all the coordinates due to the rigid connections.

**Diagonal terms** (self-inertia of each coordinate):
- `M[0,0]` = total mass of everything (cart + all links)
- `M[1,1]` = effective inertia for rotating link 1 while keeping everything else still

**Off-diagonal terms** (coupling): when you push the cart, the connected links are dragged along; when link 1 rotates, it pushes/pulls the cart. These coupling terms are products of the shared masses and cosines of the relative angles between links.

Key constants used in M (precomputed in `__post_init__`):

| Symbol | Formula | Meaning |
|--------|---------|---------|
| `α₁` | `m₁·(l₁/2) + (m₂+m₃)·l₁` | effective moment arm of link 1 |
| `α₂` | `m₂·(l₂/2) + m₃·l₂` | effective moment arm of link 2 |
| `α₃` | `m₃·(l₃/2)` | effective moment arm of link 3 |
| `β₁₂` | `(m₂·(l₂/2) + m₃·l₂)·l₁` | coupling between links 1 and 2 |
| `β₁₃` | `m₃·(l₃/2)·l₁` | coupling between links 1 and 3 |
| `β₂₃` | `m₃·(l₃/2)·l₂` | coupling between links 2 and 3 |

### Coriolis and centrifugal terms

When M(q) changes with the angles (because the cosines in the off-diagonal terms change as the links rotate), there is an extra force — similar to the centrifugal force you feel on a spinning roundabout. These are called **Coriolis and centrifugal terms** `h(q, q̇)`.

Physically: when link 2 is spinning fast, it "flings" link 1 sideways. The `h` vector captures this effect. It is proportional to the **square of the angular velocities** (hence `h ∝ θ̇²`), so it is small when everything is moving slowly and large during fast swings.

At the upright equilibrium where the controller tries to keep things, all angles are near zero and all velocities are small, so `h ≈ 0` — this is why the linearised model used for LQR design ignores `h`.

### Gravity vector G(q)

```
G = ∂V/∂q = [0, −α₁·g·sin θ₁, −α₂·g·sin θ₂, −α₃·g·sin θ₃]
```

The first entry is zero because the potential energy does not depend on cart position `x` (gravity is vertical, cart motion is horizontal).

The remaining entries tell us the gravitational torque trying to pull each link away from vertical. Key insight:

- When `θ₁ > 0` (link leans right): `G[1] = −α₁·g·sin θ₁ < 0`.  
  In the EOM: `M·q̈ = Q − G`, so the gravity term **adds** to the angular acceleration, making link 1 fall further right. **This is the instability.**
- When `θ₁ = 0` (perfectly upright): `G[1] = 0`. Gravity does nothing — this is the equilibrium.

### Putting it together

The simulation solves this system every timestep:

```
M(q) · q̈  =  [F_cart, 0, 0, 0]  −  h(q, q̇)  −  G(q)  −  D
```

Rearranging:
```
q̈ = M⁻¹ · ( [F_cart, 0, 0, 0] − h − G − D )
```

`numpy.linalg.solve` inverts the mass matrix efficiently at each timestep without explicitly forming M⁻¹.  The result `q̈ = [ẍ, θ̈₁, θ̈₂, θ̈₃]` gives the accelerations.

The ODE integrator (`scipy.integrate.solve_ivp`, RK45 method) then steps the velocities and positions forward in time:
```
q̇(t + dt) ≈ q̇(t) + q̈(t) · dt
q(t + dt)  ≈ q(t)  + q̇(t) · dt
```

### Damping

Two damping terms are included:

| Term | Formula | Physics |
|------|---------|---------|
| Cart friction | `D[0] = cart_friction · ẋ` | Rail sliding resistance (N·s/m) |
| Joint damping | `D[i] = joint_damping · θ̇ᵢ` | Bearing friction at each pivot (N·m·s/rad) |

Both oppose motion (negative sign in the EOM). Set them to zero for a perfectly frictionless simulation.

---

## Motor model

The cart is driven by a **brushed DC motor** (1120 RPM free-run speed, 2.8 N·m stall torque) connected to a **lead screw** (a threaded rod that converts rotation into linear motion).

### DC motor torque-speed curve

A brushed DC motor obeys a simple linear law:

```
τ = τ_stall · (V/V_max  −  ω_shaft / ω_free)
```

| Symbol | Meaning |
|--------|---------|
| `τ` | Shaft torque produced |
| `τ_stall` | Maximum torque (at zero speed, full voltage) |
| `V/V_max` | Normalised applied voltage = PWM duty cycle, in [−1, +1] |
| `ω_shaft` | Current shaft angular speed (from cart velocity via lead screw) |
| `ω_free` | Free-run speed (no-load, full voltage) |

At **stall** (ω_shaft = 0, full voltage): `τ = τ_stall` — maximum torque.  
At **free run** (no load): `τ = 0`, `ω_shaft = ω_free` — maximum speed.  
The **back-EMF** from the moving cart (via the lead screw) appears as the `ω_shaft / ω_free` term — the faster the cart moves, the less torque the motor can deliver.

### Lead screw conversion

```
F_cart  =  τ · (2π / lead_m) · η
```

| Symbol | Meaning |
|--------|---------|
| `lead_m` | Lead of the screw in metres/revolution (default: 8 mm = 0.008 m) |
| `η` | Mechanical efficiency (accounts for thread friction, default 0.85) |

The factor `2π / lead_m` converts rotational torque [N·m] to linear force [N].

### Primary motor (1120 RPM)

| Property | Value |
|----------|-------|
| Free-run speed | 1120 RPM |
| Stall torque | 2.8 N·m |
| Max cart speed (8 mm lead) | **0.149 m/s** |
| Stall force (8 mm lead, η=0.85) | **~1870 N** |

### Secondary motor (312 RPM)

| Property | Value |
|----------|-------|
| Free-run speed | 312 RPM |
| Stall torque | 5.2 N·m |
| Max cart speed (8 mm lead) | **0.042 m/s** |
| Stall force (8 mm lead, η=0.85) | **~3470 N** |

> **Actuator bandwidth vs. stabilisation**  
> The 8 mm lead screw limits cart speed to ~0.15 m/s. For this pendulum geometry, the LQR controller requires cart speeds up to ~0.3 m/s to stabilise a 3° initial perturbation. Therefore, the realistic motor model (with back-EMF) alone cannot keep the pendulum upright — a larger-lead screw (e.g. 20–25 mm) or a direct-drive motor is needed for a real hardware build. The simulation demonstrates ideal LQR control (motor model bypassed) so you can focus on learning the control theory; a comment in each config function shows how to enable the motor model and observe the force-saturation effect.

---

## Control

### PID controller

A **Proportional-Integral-Derivative (PID)** controller is included for
experimentation. It computes the cart force as a weighted sum of errors:

```
F_cart  =  kp_x · (x_ref − x)   −  kd_x · ẋ          (cart position)
         +  kp_θ₁ · θ₁  +  kd_θ₁ · θ̇₁                (link 1 angle)
         +  kp_θ₂ · θ₂  +  kd_θ₂ · θ̇₂                (link 2 angle)
         +  kp_θ₃ · θ₃  +  kd_θ₃ · θ̇₃                (link 3 angle)
         +  integral terms  (optional, keep small)
```

**Sign convention for the angle terms:**  
When link 1 leans right (`θ₁ > 0`), the cart must be pushed **right** (`F > 0`) to get under the pendulum's centre of mass. Therefore the P gain uses `+kp · θ` (not `−kp · θ` as in a normal setpoint controller).

**Why PD alone is hard to tune for a triple pendulum:**  
A triple pendulum has four unstable modes with complex eigenvalues. Choosing PD gains independently for each link ignores the cross-coupling between them. It is easy to accidentally choose gains that satisfy the single-link stability condition but destabilise the coupled system. The `PIDController` class is available for experimentation but the **LQR controller is recommended** for reliable stabilisation.

### LQR controller

**Linear Quadratic Regulator (LQR)** is the standard engineering solution for linear unstable systems. It automatically computes the optimal feedback gain matrix by:

1. **Linearising the physics** about the upright equilibrium (all angles and velocities zero).  
   At θ = 0: `sin θ ≈ θ`, `cos θ ≈ 1`, Coriolis terms ≈ 0.  
   The result is a linear state-space model: `ż = A·z + B·u`.

2. **Choosing cost weights** via two matrices:  
   - `Q` (8×8 diagonal): how much do you care about each state deviation?  
     Large `Q[1]` = care a lot about link 1 angle → controller corrects it faster.  
   - `R` (scalar): how much do you care about using force?  
     Large `R` = gentler controller (smaller forces, may not stabilise).

3. **Solving the algebraic Riccati equation** (via `scipy.linalg.solve_continuous_are`):  
   ```
   Aᵀ·P + P·A − P·B·R⁻¹·Bᵀ·P + Q = 0
   ```
   This gives the matrix `P`, and the optimal gain is:
   ```
   K = R⁻¹ · Bᵀ · P       (shape 1×8)
   ```

4. **Applying the control law** at every timestep:
   ```
   F_cart = −K · state
   ```

All eight gains (cart position, three angles, cart velocity, three angular velocities) are computed simultaneously, automatically accounting for the coupling between the links. This is why LQR works where hand-tuned PD fails.

**Checking stability:** after computing K, the simulation prints the eigenvalues of the closed-loop matrix `A − B·K`. All eigenvalues must have **negative real parts** (shown with ✓) for the system to be stable.

---

## Code structure

The simulation is a single file `triple_pendulum_simulation.py` divided into labelled sections:

| Section | Class / content | Purpose |
|---------|-----------------|---------|
| 1 | `SystemParameters` | All physical constants (editable) |
| 2 | `MotorModel` | DC motor + lead-screw force calculation |
| 3 | `TriplePendulumPhysics` | Mass matrix, Coriolis, gravity, EOM solver |
| 4 | `PIDController` | Manual PD/PID controller (for experimentation) |
| 4b | `LQRController` | Automatic optimal LQR controller |
| 5 | `Simulation` | Zero-Order-Hold ODE integrator |
| 6 | `Visualizer` | Matplotlib animation and time-series plots |
| 7 | Config functions | Four ready-to-run test setups |
| 8 | `main()` | Entry point; runs and visualises a named config |

---

## How to change parameters

### Change link geometry

Edit `SystemParameters` in `config_free_swing()` or create your own config function:

```python
params = SystemParameters(
    M_cart = 2.0,     # Cart mass [kg]
    l1 = 0.30,        # Link 1 length [m]  ← change this
    m1 = 0.15,        # Link 1 mass [kg]   ← and this
    l2 = 0.25,        # Link 2 length [m]
    m2 = 0.10,
    l3 = 0.20,        # Link 3 length [m]
    m3 = 0.08,
    x_min = -0.5,     # Cart travel limits [m]
    x_max =  0.5,
    cart_friction   = 5.0,   # Rail friction [N·s/m]
    joint_damping   = 0.001, # Pivot friction [N·m·s/rad]
)
```

After changing `params`, LQR gains are re-computed automatically.

### Change motor

```python
# Primary motor (1120 RPM) — default
motor = MotorModel(free_speed_rpm=1120.0, stall_torque_Nm=2.8, lead_mm=8.0)

# Secondary motor (312 RPM)
motor = MotorModel.secondary_motor()

# Custom motor with larger lead screw (faster cart speed)
motor = MotorModel(free_speed_rpm=1120.0, stall_torque_Nm=2.8, lead_mm=25.0)
```

### Change LQR aggressiveness

```python
controller = LQRController(
    physics=physics,
    Q_diag=[10.0, 2000.0, 2000.0, 2000.0, 1.0, 100.0, 100.0, 100.0],
    #        x     θ₁      θ₂      θ₃      ẋ    θ̇₁     θ̇₂     θ̇₃
    R=0.001,        # smaller → more aggressive force
    max_force=600,  # force clamp [N]
)
```

Increase `Q_diag[0]` to keep the cart nearer to the centre. Increase `Q_diag[1..3]` to correct angle deviations faster. Increase `R` to see how a weaker controller (limited force) fails.

### Add a new configuration

```python
def config_my_experiment():
    params = SystemParameters(M_cart=2.5, l1=0.4, m1=0.2, ...)
    motor  = MotorModel(free_speed_rpm=1120)
    phys   = TriplePendulumPhysics(params)
    ctrl   = LQRController(phys, R=0.005)
    state0 = np.array([0, np.radians(5), 0, 0, 0, 0, 0, 0])
    return params, motor, ctrl, state0, False   # False = ideal force

CONFIGURATIONS["my_experiment"] = config_my_experiment
```

Then run:
```bash
python triple_pendulum_simulation.py --config my_experiment
```

---

## Configurations

| Name | Controller | Motor model | Description |
|------|-----------|-------------|-------------|
| `free_swing` | None | No | Pendulum falls freely; verifies physics |
| `pd_stabilise` | LQR | No (ideal) | Optimal stabilisation, default geometry |
| `longer_links` | LQR | No (ideal) | Longer heavier links; LQR adapts |
| `secondary_motor` | LQR | No (ideal) | Heavier cart, slower motor info shown |

> **Tip:** change `False` → `True` in the `return` statement of any LQR config to enable the motor model and observe how the back-EMF limits the achievable stabilisation.

---

## Output and visualisation

### Command-line options

```
python triple_pendulum_simulation.py
    --config       pd_stabilise    # which configuration to run
    --t_end        5.0             # simulation duration [s]
    --dt           0.01            # control timestep [s]
    --no-animate                   # skip animation (just plot)
    --no-show                      # do not call plt.show()
    --save-plot    plot.png        # save time-series figure
    --save-anim    anim.gif        # save animation as GIF
```

### Time-series plots (4 panels)

1. **Angles** — θ₁, θ₂, θ₃ over time [degrees]
2. **Angular velocities** — θ̇₁, θ̇₂, θ̇₃ [degrees/s]
3. **Cart position and velocity** — x [m] and ẋ [m/s]
4. **Control force** — F_cart [N] applied to the cart

### Animation

A real-time Matplotlib animation shows the cart sliding along the rail and the three links swinging. Red dashed vertical lines mark the travel limits.

### Programmatic use

```python
from triple_pendulum_simulation import *
import numpy as np

params = SystemParameters(M_cart=2.0, l1=0.3, m1=0.15,
                          l2=0.25, m2=0.1, l3=0.2, m3=0.08)
physics = TriplePendulumPhysics(params)
ctrl    = LQRController(physics, R=0.001)
motor   = MotorModel()

sim    = Simulation(physics, ctrl, motor)
result = sim.run((0, 5), np.radians([0, 3, 2, 1.5, 0, 0, 0, 0]),
                 use_motor_model=False)

print(result.theta1)   # link 1 angle time series [rad]
print(result.x)        # cart position time series [m]
print(result.force)    # applied force time series [N]
```

---

## FAQ

**Q: Why does the pendulum fall even with the controller on?**  
A: If you enable the motor model (`use_motor_model=True`) the cart speed is capped by the motor's free-run speed. With an 8 mm lead screw, that is only ~0.15 m/s — too slow to catch a falling triple pendulum from a 3° initial lean. Use a larger lead screw (20–25 mm) or bypass the motor model to see successful LQR stabilisation.

**Q: Why does the LQR only work near the upright position?**  
A: LQR is designed on the *linearised* model (valid only for small angles). For large angles the nonlinear terms (sin θ instead of θ, Coriolis forces, etc.) become significant and the linear gain matrix K is no longer optimal. To stabilise from large angles you would first need a swing-up manoeuvre (a separate nonlinear control problem) and then hand off to the LQR when close to upright.

**Q: How do I interpret the closed-loop eigenvalues?**  
A: Each eigenvalue `λ = a ± bj` describes a mode of the closed-loop system. `a < 0` (negative real part) means that mode decays — it is stable. `b ≠ 0` means the decay is oscillatory (like a spring). The eigenvalue with the smallest `|a|` is the "slowest" mode and determines how long settling takes.

**Q: What is the Zero-Order Hold (ZOH) integration strategy?**  
A: Rather than evaluating the controller inside the ODE solver (which causes issues because the controller has internal state), we split time into discrete intervals. At the start of each interval, the controller evaluates the current state and outputs a constant force. The ODE is then integrated for that interval with the force held constant ("zero-order hold"). This makes the ODE function a pure function of `(t, state)` — no side effects — so the adaptive step-size solver works correctly.

**Q: Can I use a different number of links?**  
A: Not without modifying the physics engine, which is hardcoded for three links. To extend to N links, the mass matrix, Coriolis, and gravity vectors would need to be generalised (e.g. with a loop over N, or using symbolic derivation with SymPy).

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.21 | Array maths, linear algebra |
| `scipy` | ≥ 1.7  | ODE integration, Riccati equation solver |
| `matplotlib` | ≥ 3.4 | Plots and animation |

All are standard scientific Python packages available via `pip` or `conda`.
