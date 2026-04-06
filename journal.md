# Project Journal

## Project Overview

This repository is for a triple inverted pendulum project built around a real CAD-designed mechanism and an Isaac Sim / Isaac Lab reinforcement learning workflow.

The high-level goal is:
- use the CAD-exported URDF of the actual mechanism
- make the mechanism physics in Isaac match the real hardware behavior as closely as possible
- train policies that can:
  - swing the triple pendulum up from the hanging state
  - balance it near the upright state
  - eventually do both in one combined policy

The original repo also contains earlier classical dynamics / controls work for the triple inverted pendulum, but the active work tracked in this journal is the Isaac Sim / Isaac Lab path using the uploaded CAD-exported robot model.

## Actual Mechanism / CAD Notes

The physical system represented by the CAD and URDF is not just a generic cart-pole. It is a motor-driven linear stage that moves the pendulum base.

Important intended mechanism behavior:
- the two motor mates are the active drive inputs
- the motor drive turns / drives the screw mechanism
- the screw mechanism moves the carriage / head that the pendulum is mounted on
- the pendulum linkage revolute joints are supposed to be passive, dead, or hanging under gravity rather than actively actuated

This distinction matters because directly actuating a cart or carriage joint would not match the actual CAD mechanism the user wants modeled.

## Current Project Objective

Before trusting any reinforcement learning result, the mechanism physics need to be checked against the intended CAD behavior.

Current priority order:
1. make sure the modeled mechanism behaves like the actual CAD assembly
2. verify that the motor-driven translation and passive hanging pendulum links are represented correctly
3. only then continue staged RL for:
   - balance
   - swing-up
   - combined swing-up + balance

## Journal Scope

This journal tracks the work done on the triple inverted pendulum Isaac Sim / Isaac Lab workflow.

Notes:
- Entries before exact script/log timestamps were reconstructed from the session and are marked as approximate.
- Times use Pacific time where known.

## 2026-04-05

### 2026-04-05 20:30 -07:00 (approx)
- Reviewed the original repository goal.
- Confirmed the repo started as a triple inverted pendulum dynamics / control project with classical simulation and controller work.
- Identified the uploaded CAD-exported URDF zip as the asset to use for reinforcement learning in Isaac.

### 2026-04-05 20:50 -07:00 (approx)
- Extracted and inspected the uploaded URDF and mesh assets.
- Identified the useful task joints:
  - cart axis: `planar_1_2`
  - pendulum joints: `revolute_1_0`, `revolute_2_0`, `revolute_3_0`
- Identified extra CAD drivetrain / assembly joints that needed to be locked for RL.

### 2026-04-05 21:10 -07:00 (approx)
- Added an external Isaac Lab task package under `source/triple_pendulum_isaac`.
- Implemented three staged tasks:
  - `TriplePendulum-Balance-Direct-v0`
  - `TriplePendulum-SwingUp-Direct-v0`
  - `TriplePendulum-SwingUpBalance-Direct-v0`
- Configured the mechanism to spawn elevated on a table so the hanging links do not hit the ground.

### 2026-04-05 21:25 -07:00 (approx)
- Added helper scripts under `isaaclab/` for:
  - environment listing
  - SB3 training
  - SB3 playback
  - URDF asset sanitization
  - one-shot staged training
- Added package install compatibility files so editable install works reliably.

### 2026-04-05 21:40 -07:00 (approx)
- Fixed the package metadata so the task package no longer referenced a README outside its own package root.
- Added a local package README and `setup.py` fallback for older editable-install behavior.

### 2026-04-05 22:00 -07:00 (approx)
- Resolved task registration visibility by adding wrapper launcher scripts that import the external task package before calling the stock Isaac Lab SB3 scripts.
- Verified the three custom `TriplePendulum-*` environments appeared in the Gym registry.

## 2026-04-06

### 2026-04-06 08:30 -07:00 (approx)
- Sanitized the CAD-exported URDF asset names and mesh filenames to avoid Isaac / USD import failures caused by invalid prim names and mesh references.
- Switched the environment to load the sanitized URDF.

### 2026-04-06 08:45 -07:00 (approx)
- Fixed the PPO config collision by removing the duplicate `verbose` setting from the SB3 YAML.
- Verified that the balance task could complete a one-iteration training bootstrap and write a checkpoint.

### 2026-04-06 08:55 -07:00 (approx)
- Verified the swing-up task could also complete a one-iteration bootstrap and write a checkpoint.
- Confirmed the combined task also trained when run on its own rather than in a conflicting parallel Isaac Sim launch.

### 2026-04-06 09:00 -07:00 (approx)
- Reworked `isaaclab/run_all.ps1` to use the wrapper-based training flow that actually works with this project.
- Added stage sequencing, pauses between stages, and checkpoint existence checks.

### 2026-04-06 09:05 -07:00 (approx)
- Tuned the run settings for the user's laptop GPU:
  - `64` envs recommended for headless unattended training on an `8 GB` RTX 5060 Laptop GPU
  - `8` envs recommended for visible windowed training
  - smaller env counts recommended for cleaner viewing
- Reduced PPO `batch_size` to `4096` to better fit the `64`-env rollout size.

### 2026-04-06 09:16 -07:00
- Created this `journal.md` file at the repo root.
- Added reconstructed entries for the work completed so far.
- Starting from this point, the journal is the running record for this project in the repository.

### 2026-04-06 09:20 -07:00 (approx)
- User requested that the repo maintain an ongoing dated journal for the project.
- Confirmed the journal will be updated going forward as changes are made.

### 2026-04-06 09:35 -07:00 (approx)
- Hardened `isaaclab/run_all.ps1` for unattended use.
- Added staged execution, pauses between stages, retries, and checkpoint verification so the script only reports success when `model.zip` files are actually produced.

### 2026-04-06 09:40 -07:00 (approx)
- Verified the one-shot training workflow end-to-end on the local machine with the staged Isaac Lab tasks.
- Confirmed the script can complete balance, swing-up, and combined runs sequentially and detect the final checkpoints.

### 2026-04-06 09:45 -07:00 (approx)
- User tested visible training and expected a more video-like Isaac Sim view.
- Clarified that training output in the terminal is normal and that `play_sb3.py` is the correct mode for a clean visual demo after training.

### 2026-04-06 09:55 -07:00 (approx)
- User encountered a windowed training crash with an `h5py` DLL load trace during Isaac Lab startup.
- Re-checked the Isaac Sim Python environment and confirmed `h5py` imports successfully when tested directly, so the environment is not currently missing the package.
- Treated the issue as workflow-sensitive rather than a permanent package absence.

### 2026-04-06 10:00 -07:00 (approx)
- Tuned the SB3 PPO minibatch setting again for laptop-scale visible runs.
- Changed `batch_size` from `4096` to `512` in the PPO config so `8`-env runs no longer generate the truncated minibatch warning and remain compatible with the `64`-env headless workflow.

### 2026-04-06 09:20 -07:00
- Diagnosed the later Isaac Lab startup crash as an eager `h5py` import path inside Isaac Lab's dataset/recorder utilities, not as a failure of the triple-pendulum task itself.
- Patched `C:\Users\emmad\Downloads\IsaacLab\source\isaaclab\isaaclab\utils\datasets\hdf5_dataset_file_handler.py` to load `h5py` lazily so training/play startup no longer depends on HDF5 unless dataset export is actually used.

### 2026-04-06 09:21 -07:00
- Repaired the Isaac Sim bundled Python environment after an incompatible reinstall had upgraded NumPy to `2.4.4`.
- Restored Isaac-Lab-compatible versions in `C:\Users\emmad\Downloads\IsaacLab\_isaac_sim`:
  - `numpy==1.26.4`
  - `h5py==3.12.1`

### 2026-04-06 09:24 -07:00
- Reinstalled the editable `triple_pendulum_isaac` package into Isaac Sim's bundled Python.
- Re-verified custom Gym registrations:
  - `TriplePendulum-Balance-Direct-v0`
  - `TriplePendulum-SwingUp-Direct-v0`
  - `TriplePendulum-SwingUpBalance-Direct-v0`

### 2026-04-06 09:24 -07:00
- Verified the previously failing combined-policy playback path now starts successfully through Isaac Lab.
- Successfully ran:
  - `play_sb3.py`
  - task: `TriplePendulum-SwingUpBalance-Direct-v0`
  - checkpoint: `C:\Users\emmad\Downloads\IsaacLab\logs\sb3\TriplePendulum-SwingUpBalance-Direct-v0\2026-04-05_21-38-03\model.zip`
  - `num_envs=1`
  - headless video capture mode for a bounded smoke test

### 2026-04-06 09:40 -07:00
- Reviewed the exported URDF joint tree against the user's mechanism description.
- Confirmed the current task had been directly actuating the exported carriage prismatic joint `planar_1_2`, which bypassed the motor semantics the user expected.
- Confirmed the passive hanging linkage joints are:
  - `revolute_1_0`
  - `revolute_2_0`
  - `revolute_3_0`
- Confirmed several CAD-exported joints (`axle_*`, `revolute_1_1`, `revolute_1_2`, `cylindrical_*`) are not a clean closed-loop transmission in URDF form and need explicit handling.

### 2026-04-06 09:42 -07:00
- Reworked the Isaac Lab env actuation model in `source/triple_pendulum_isaac/triple_pendulum_isaac/tasks/direct/triple_inverted_pendulum/env.py`.
- New mechanism behavior:
  - RL action commands an internal motor state instead of directly forcing the carriage.
  - The two motor joints `axle_0` and `axle_1` are now the commanded motor mates.
  - Carriage translation `planar_1_2` is derived from motor angle using the lead screw lead (`8 mm / rev`).
  - The passive linkage revolutes remain unactuated so they can hang freely under gravity.
  - Redundant exported planar/screw/support joints remain locked so the URDF's broken open-chain version of the CAD transmission does not corrupt the pendulum physics.

### 2026-04-06 09:47 -07:00
- Added `isaaclab/inspect_mechanism.py` to drive the mechanism with an open-loop sinusoidal motor command for visual inspection without mixing in policy behavior.
- Verified the updated environment still boots and can complete a one-iteration SB3 balance run after the actuation-model change.

### 2026-04-06 09:53 -07:00
- Troubleshot a failed `inspect_mechanism.py` launch from PowerShell.
- Confirmed the script itself exists and is valid at:
  - `C:\Users\emmad\Downloads\CodeP\Triple-Inverted-Pedulum\isaaclab\inspect_mechanism.py`
- Confirmed the user's failing command used the wrong folder spelling (`Triple-Inverted-Pendulum` instead of `Triple-Inverted-Pedulum`).
- Next guidance is to rerun the same inspection command with the corrected `Pedulum` path before changing any physics or reinstalling Isaac Sim / Isaac Lab.

### 2026-04-06 09:56 -07:00
- User clarified additional real-mechanism behavior that the Isaac model should respect.
- Required physical interpretation going forward:
  - the horizontal motor turning should move the carriage along the global `x` axis
  - the roll axis should also move as part of the driven mechanism behavior
  - the pendulum linkage joints should stay passive and hang / swing under physics rather than being directly driven
- This clarification supersedes any simpler cart-only interpretation of the CAD export.

### 2026-04-06 09:58 -07:00
- Re-checked the sanitized URDF joint definitions against the user's latest mechanism description.
- Confirmed likely exported joint roles:
  - `planar_1_2`: prismatic axis `+x` translation
  - `planar_1_4`: continuous roll axis about local `z`
  - `cylindrical_1_1` / `cylindrical_1_2`: lead-screw-related prismatic / rotary pair
  - `revolute_1_0`, `revolute_2_0`, `revolute_3_0`: passive pendulum linkage joints
- Conclusion:
  - the simulation should not be treated as pure 1-DOF cart motion
  - the driven mechanism likely needs both `x` translation and roll behavior represented
  - the hanging linkage joints should remain physics-driven and unactuated
