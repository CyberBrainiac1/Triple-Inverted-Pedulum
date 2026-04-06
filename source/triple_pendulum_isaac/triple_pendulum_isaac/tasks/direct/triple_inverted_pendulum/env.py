from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


REPO_ROOT = Path(__file__).resolve().parents[6]
URDF_PATH = (
    REPO_ROOT / "_sanitized_urdf" / "finaltripleinvertedpendulum" / "urdf" / "finaltripleinvertedpendulum_sanitized.urdf"
)


@configclass
class TripleInvertedPendulumEnvCfg(DirectRLEnvCfg):
    """Direct RL task base config for the CAD-exported triple inverted pendulum."""

    decimation = 2
    episode_length_s = 20.0
    action_scale = 1.0
    action_space = 1
    observation_space = 12
    state_space = 0
    lead_screw_lead = 0.008
    motor_accel_scale = 180.0
    max_motor_speed = 60.0
    motor_damping = 8.0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(URDF_PATH),
            fix_base=True,
            merge_fixed_joints=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None,
                    damping=None,
                )
            ),
            self_collision=False,
            collision_from_visuals=False,
            collider_type="convex_hull",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.86),
            joint_pos={
                "planar_1_2": 0.0,
                "planar_1_3": 0.0,
                "planar_1_4": 0.0,
                "revolute_1_0": 0.0,
                "revolute_2_0": 0.0,
                "revolute_3_0": 0.0,
                "revolute_1_1": 0.0,
                "revolute_1_2": 0.0,
                "cylindrical_1_1": 0.0,
                "cylindrical_1_2": 0.0,
                "axle_0": 0.0,
                "axle_1": 0.0,
            },
        ),
        actuators={
            "carriage": ImplicitActuatorCfg(
                joint_names_expr=["planar_1_2"],
                stiffness=20000.0,
                damping=2500.0,
                effort_limit_sim=100000.0,
            ),
            "motors": ImplicitActuatorCfg(
                joint_names_expr=["axle_0", "axle_1"],
                stiffness=250.0,
                damping=25.0,
                effort_limit_sim=5000.0,
            ),
            "locks": ImplicitActuatorCfg(
                joint_names_expr=[
                    "planar_1_3",
                    "planar_1_4",
                    "cylindrical_1_1",
                    "cylindrical_1_2",
                    "revolute_1_1",
                    "revolute_1_2",
                ],
                stiffness=10000.0,
                damping=1000.0,
                effort_limit_sim=100000.0,
            ),
        },
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    table_size = (0.8, 0.8, 0.8)
    table_center = (0.0, 0.0, 0.4)

    cart_dof_name = "planar_1_2"
    motor_dof_names = ["axle_0", "axle_1"]
    pendulum_dof_names = ["revolute_1_0", "revolute_2_0", "revolute_3_0"]
    locked_dof_names = [
        "planar_1_3",
        "planar_1_4",
        "cylindrical_1_1",
        "cylindrical_1_2",
        "revolute_1_1",
        "revolute_1_2",
    ]

    max_cart_pos = 0.14
    initial_cart_pos_range = (-0.02, 0.02)
    initial_cart_vel_range = (-0.05, 0.05)
    initial_pendulum_hanging_range = (-0.20, 0.20)
    initial_pendulum_upright_range = (-0.10, 0.10)
    initial_pendulum_full_range = (-math.pi, math.pi)
    initial_pendulum_vel_range = (-0.25, 0.25)

    reset_hanging_ratio = 0.60
    reset_upright_ratio = 0.20
    terminate_on_large_angle = False
    max_pendulum_angle = math.pi / 2

    balance_capture_angle = 0.20
    balance_capture_vel = 1.50

    rew_scale_alive = 0.25
    rew_scale_height = 2.0
    rew_scale_balance = 4.0
    rew_scale_balance_bonus = 3.0
    rew_scale_cart_pos = 1.0
    rew_scale_cart_vel = 0.02
    rew_scale_joint_vel = 0.01
    rew_scale_action_rate = 0.002
    rew_scale_terminated = -5.0


@configclass
class TripleInvertedPendulumBalanceEnvCfg(TripleInvertedPendulumEnvCfg):
    episode_length_s = 10.0
    motor_accel_scale = 120.0
    reset_hanging_ratio = 0.0
    reset_upright_ratio = 1.0
    terminate_on_large_angle = True
    rew_scale_alive = 1.0
    rew_scale_height = 0.5
    rew_scale_balance = 6.0
    rew_scale_balance_bonus = 4.0


@configclass
class TripleInvertedPendulumSwingUpEnvCfg(TripleInvertedPendulumEnvCfg):
    episode_length_s = 20.0
    motor_accel_scale = 220.0
    reset_hanging_ratio = 1.0
    reset_upright_ratio = 0.0
    rew_scale_height = 2.5
    rew_scale_balance = 2.0
    rew_scale_balance_bonus = 2.5


@configclass
class TripleInvertedPendulumSwingUpBalanceEnvCfg(TripleInvertedPendulumEnvCfg):
    episode_length_s = 20.0
    motor_accel_scale = 180.0
    reset_hanging_ratio = 0.60
    reset_upright_ratio = 0.20


class TripleInvertedPendulumEnv(DirectRLEnv):
    cfg: TripleInvertedPendulumEnvCfg

    def __init__(self, cfg: TripleInvertedPendulumEnvCfg, render_mode: str | None = None, **kwargs):
        if not URDF_PATH.exists():
            raise FileNotFoundError(
                f"Expected extracted URDF at '{URDF_PATH}'. Extract FinalTripleInvertedPendulum.zip before launching."
            )
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.robot.find_joints(self.cfg.cart_dof_name)
        self._motor_dof_idx, _ = self.robot.find_joints(self.cfg.motor_dof_names)
        self._pendulum_dof_idx, _ = self.robot.find_joints(self.cfg.pendulum_dof_names)
        self._locked_dof_idx, _ = self.robot.find_joints(self.cfg.locked_dof_names)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.motor_pos = torch.zeros_like(self.actions)
        self.motor_vel = torch.zeros_like(self.actions)
        self._meters_per_rad = self.cfg.lead_screw_lead / (2.0 * math.pi)
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane("/World/ground", GroundPlaneCfg())
        table_cfg = sim_utils.CuboidCfg(
            size=self.cfg.table_size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        table_cfg.func("/World/envs/env_0/Table", table_cfg, translation=self.cfg.table_center)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self) -> None:
        # The CAD export lost the original motor-to-screw closed loop, so we drive the two
        # motor mates and convert their averaged angle into carriage motion through the screw lead.
        motor_accel = self.actions * self.cfg.motor_accel_scale - self.cfg.motor_damping * self.motor_vel
        self.motor_vel = torch.clamp(
            self.motor_vel + self.step_dt * motor_accel,
            -self.cfg.max_motor_speed,
            self.cfg.max_motor_speed,
        )
        self.motor_pos = self.motor_pos + self.step_dt * self.motor_vel

        cart_pos = torch.clamp(self.motor_pos * self._meters_per_rad, -self.cfg.max_cart_pos, self.cfg.max_cart_pos)
        clipped = torch.abs(cart_pos) >= (self.cfg.max_cart_pos - 1.0e-6)
        self.motor_pos = cart_pos / self._meters_per_rad
        self.motor_vel = torch.where(clipped, torch.zeros_like(self.motor_vel), self.motor_vel)

        self.robot.set_joint_position_target(cart_pos, joint_ids=self._cart_dof_idx)
        self.robot.set_joint_velocity_target(self.motor_vel * self._meters_per_rad, joint_ids=self._cart_dof_idx)

        motor_targets = self.motor_pos.repeat(1, len(self._motor_dof_idx))
        motor_vel_targets = self.motor_vel.repeat(1, len(self._motor_dof_idx))
        self.robot.set_joint_position_target(motor_targets, joint_ids=self._motor_dof_idx)
        self.robot.set_joint_velocity_target(motor_vel_targets, joint_ids=self._motor_dof_idx)

    def _get_observations(self) -> dict:
        cart_pos = self.joint_pos[:, self._cart_dof_idx]
        cart_vel = self.joint_vel[:, self._cart_dof_idx]
        pend_pos = self.joint_pos[:, self._pendulum_dof_idx]
        pend_vel = self.joint_vel[:, self._pendulum_dof_idx]

        obs = torch.cat(
            (
                cart_pos,
                cart_vel,
                torch.sin(pend_pos),
                torch.cos(pend_pos),
                pend_vel,
                self.actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        cart_pos = self.joint_pos[:, self._cart_dof_idx[0]]
        cart_vel = self.joint_vel[:, self._cart_dof_idx[0]]
        pend_pos = self.joint_pos[:, self._pendulum_dof_idx]
        pend_vel = self.joint_vel[:, self._pendulum_dof_idx]
        action_delta = (self.actions - self.prev_actions).squeeze(-1)

        return compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_height,
            self.cfg.rew_scale_balance,
            self.cfg.rew_scale_balance_bonus,
            self.cfg.rew_scale_cart_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_joint_vel,
            self.cfg.rew_scale_action_rate,
            self.cfg.rew_scale_terminated,
            self.cfg.balance_capture_angle,
            self.cfg.balance_capture_vel,
            cart_pos,
            cart_vel,
            pend_pos,
            pend_vel,
            action_delta,
            self.reset_terminated,
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        cart_out = torch.abs(self.joint_pos[:, self._cart_dof_idx[0]]) > self.cfg.max_cart_pos
        if self.cfg.terminate_on_large_angle:
            angle_out = torch.any(
                torch.abs(self.joint_pos[:, self._pendulum_dof_idx]) > self.cfg.max_pendulum_angle,
                dim=1,
            )
        else:
            angle_out = torch.zeros_like(cart_out)
        invalid = ~torch.isfinite(self.joint_pos).all(dim=1) | ~torch.isfinite(self.joint_vel).all(dim=1)
        terminated = cart_out | angle_out | invalid
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        num_envs = joint_pos.shape[0]
        mode_draw = torch.rand(num_envs, device=self.device)
        hanging_mask = mode_draw < self.cfg.reset_hanging_ratio
        upright_mask = (mode_draw >= self.cfg.reset_hanging_ratio) & (
            mode_draw < self.cfg.reset_hanging_ratio + self.cfg.reset_upright_ratio
        )
        full_mask = ~(hanging_mask | upright_mask)

        joint_pos[:, self._cart_dof_idx] = sample_uniform(
            self.cfg.initial_cart_pos_range[0],
            self.cfg.initial_cart_pos_range[1],
            joint_pos[:, self._cart_dof_idx].shape,
            self.device,
        )

        if torch.any(hanging_mask):
            hanging_rows = torch.nonzero(hanging_mask, as_tuple=False).squeeze(-1)
            joint_pos[hanging_rows[:, None], self._pendulum_dof_idx] = math.pi + sample_uniform(
                self.cfg.initial_pendulum_hanging_range[0],
                self.cfg.initial_pendulum_hanging_range[1],
                (hanging_rows.shape[0], len(self._pendulum_dof_idx)),
                self.device,
            )
        if torch.any(upright_mask):
            upright_rows = torch.nonzero(upright_mask, as_tuple=False).squeeze(-1)
            joint_pos[upright_rows[:, None], self._pendulum_dof_idx] = sample_uniform(
                self.cfg.initial_pendulum_upright_range[0],
                self.cfg.initial_pendulum_upright_range[1],
                (upright_rows.shape[0], len(self._pendulum_dof_idx)),
                self.device,
            )
        if torch.any(full_mask):
            full_rows = torch.nonzero(full_mask, as_tuple=False).squeeze(-1)
            joint_pos[full_rows[:, None], self._pendulum_dof_idx] = sample_uniform(
                self.cfg.initial_pendulum_full_range[0],
                self.cfg.initial_pendulum_full_range[1],
                (full_rows.shape[0], len(self._pendulum_dof_idx)),
                self.device,
            )
        joint_pos[:, self._locked_dof_idx] = 0.0
        motor_pos = joint_pos[:, self._cart_dof_idx] / self._meters_per_rad
        joint_pos[:, self._motor_dof_idx] = motor_pos.repeat(1, len(self._motor_dof_idx))

        joint_vel[:, self._cart_dof_idx] = sample_uniform(
            self.cfg.initial_cart_vel_range[0],
            self.cfg.initial_cart_vel_range[1],
            joint_vel[:, self._cart_dof_idx].shape,
            self.device,
        )
        motor_vel = torch.clamp(joint_vel[:, self._cart_dof_idx] / self._meters_per_rad, -self.cfg.max_motor_speed, self.cfg.max_motor_speed)
        joint_vel[:, self._pendulum_dof_idx] = sample_uniform(
            self.cfg.initial_pendulum_vel_range[0],
            self.cfg.initial_pendulum_vel_range[1],
            joint_vel[:, self._pendulum_dof_idx].shape,
            self.device,
        )
        joint_vel[:, self._locked_dof_idx] = 0.0
        joint_vel[:, self._motor_dof_idx] = motor_vel.repeat(1, len(self._motor_dof_idx))

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.motor_pos[env_ids] = motor_pos
        self.motor_vel[env_ids] = motor_vel

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_height: float,
    rew_scale_balance: float,
    rew_scale_balance_bonus: float,
    rew_scale_cart_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_joint_vel: float,
    rew_scale_action_rate: float,
    rew_scale_terminated: float,
    balance_capture_angle: float,
    balance_capture_vel: float,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    pend_pos: torch.Tensor,
    pend_vel: torch.Tensor,
    action_delta: torch.Tensor,
    reset_terminated: torch.Tensor,
) -> torch.Tensor:
    alive = rew_scale_alive * (1.0 - reset_terminated.float())
    terminated = rew_scale_terminated * reset_terminated.float()
    upright_cos = torch.mean(torch.cos(pend_pos), dim=-1)
    height_reward = rew_scale_height * (upright_cos + 1.0) * 0.5
    balance_reward = rew_scale_balance * torch.exp(
        -4.0 * torch.sum(torch.square(pend_pos), dim=-1) - 0.25 * torch.sum(torch.square(pend_vel), dim=-1)
    )
    in_balance = (
        torch.all(torch.abs(pend_pos) < balance_capture_angle, dim=-1)
        & torch.all(torch.abs(pend_vel) < balance_capture_vel, dim=-1)
    ).float()
    balance_bonus = rew_scale_balance_bonus * in_balance
    cart_penalty = rew_scale_cart_pos * torch.square(cart_pos)
    cart_vel_penalty = rew_scale_cart_vel * torch.abs(cart_vel)
    joint_vel_penalty = rew_scale_joint_vel * torch.sum(torch.abs(pend_vel), dim=-1)
    action_rate_penalty = rew_scale_action_rate * torch.square(action_delta)
    return (
        alive
        + height_reward
        + balance_reward
        + balance_bonus
        - cart_penalty
        - cart_vel_penalty
        - joint_vel_penalty
        - action_rate_penalty
        + terminated
    )
