# Isaac Lab Integration

This repository now includes a minimal external Isaac Lab task package that uses the uploaded CAD-exported URDF for staged reinforcement learning:

- `TriplePendulum-Balance-Direct-v0`
- `TriplePendulum-SwingUp-Direct-v0`
- `TriplePendulum-SwingUpBalance-Direct-v0`

## What this adds

- Task names:
  - `TriplePendulum-Balance-Direct-v0`
  - `TriplePendulum-SwingUp-Direct-v0`
  - `TriplePendulum-SwingUpBalance-Direct-v0`
- Workflow: Isaac Lab `DirectRLEnv`
- Asset source: `FinalTripleInvertedPendulum.zip`, extracted under [`_extracted_urdf/finaltripleinvertedpendulum/urdf/finaltripleinvertedpendulum.urdf`](../_extracted_urdf/finaltripleinvertedpendulum/urdf/finaltripleinvertedpendulum.urdf)
- RL library config included: Stable-Baselines3 PPO

The environment treats the mechanism as a cart balancing three serial pendulum joints:

- Controlled joint: `planar_1_2` (cart translation along the rail)
- Balance joints: `revolute_1_0`, `revolute_2_0`, `revolute_3_0`
- Locked nuisance joints from the CAD export: `planar_1_3`, `planar_1_4`, `revolute_1_1`, `revolute_1_2`, `cylindrical_1_1`, `cylindrical_1_2`, `axle_0`, `axle_1`

This matches the goal of the original repo and your follow-up requirement: train in stages if you want, then combine. The robot is also spawned on a raised table so the linkage can hang below the cart without striking the ground plane.

## Install inside Isaac Lab

Use a Python environment that already has Isaac Lab and Isaac Sim working.

```powershell
cd C:\Users\emmad\Downloads\CodeP\Triple-Inverted-Pedulum
python -m pip install -e .\source\triple_pendulum_isaac
```

## Sanity check the task registration

```powershell
cd <YOUR_ISAACLAB_ROOT>
isaaclab.bat -p scripts\list_envs.py
```

You should see the three `TriplePendulum-*` tasks in the registry after the package is installed.

## Train

From your Isaac Lab checkout:

Recommended order:

```powershell
cd <YOUR_ISAACLAB_ROOT>
isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task TriplePendulum-Balance-Direct-v0 --num_envs 256 --headless
isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task TriplePendulum-SwingUp-Direct-v0 --num_envs 256 --headless
isaaclab.bat -p scripts\reinforcement_learning\sb3\train.py --task TriplePendulum-SwingUpBalance-Direct-v0 --num_envs 256 --headless
```

If your GPU has more memory available, increase `--num_envs`. The default task config is set to `1024` envs when not overridden. The combined task is harder than pure balance; expect to train for longer.

## Play a trained checkpoint

```powershell
cd <YOUR_ISAACLAB_ROOT>
isaaclab.bat -p scripts\reinforcement_learning\sb3\play.py --task TriplePendulum-SwingUpBalance-Direct-v0 --checkpoint <PATH_TO_MODEL_ZIP>
```

## Notes on the CAD URDF

- The uploaded URDF is a full mechanical assembly, not a minimal academic cart-pole model.
- The task locks extra drivetrain and planar joints so RL acts on the cart axis only.
- The robot root is elevated to `z=0.86` and a table collider is spawned under it so hanging starts have clearance.
- Resets are staged: upright for balance, hanging for swing-up, and mixed for the combined task.
- The Isaac Lab docs currently recommend using `UrdfFileCfg` for direct URDF spawning or converting the URDF to USD with the official converter when you want tighter asset control.

Relevant official references:

- [Creating a Direct Workflow RL Environment](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html)
- [Registering an Environment](https://isaac-sim.github.io/IsaacLab/develop/source/tutorials/03_envs/register_rl_env_gym.html)
- [Importing a New Asset](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/import_new_asset.html)
- [Writing an Asset Configuration](https://isaac-sim.github.io/IsaacLab/develop/source/how-to/write_articulation_cfg.html)
