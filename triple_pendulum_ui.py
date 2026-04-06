from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import json
import tkinter as tk
from tkinter import ttk
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import triple_pendulum_simulation as sim


BG = "#000000"
PANEL = "#090909"
TRACK = "#1f1f1f"
CART = "#d7d7d7"
LINK_1 = "#ffffff"
LINK_2 = "#cfcfcf"
LINK_3 = "#9f9f9f"
JOINT = "#f5a623"
TEXT = "#f2f2f2"
MUTED = "#8a8a8a"

REPO_ROOT = Path(__file__).resolve().parent
V2_URDF_PATH = (
    REPO_ROOT
    / "_extracted_urdf_v2"
    / "finaltripleinvertedpendulum"
    / "urdf"
    / "finaltripleinvertedpendulum.urdf"
)


def _parse_vec3(value: str) -> np.ndarray:
    return np.array([float(part) for part in value.split()], dtype=float)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _origin_transform(element: ET.Element) -> np.ndarray:
    xyz = _parse_vec3(element.attrib.get("xyz", "0 0 0"))
    rpy = _parse_vec3(element.attrib.get("rpy", "0 0 0"))
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _gltf_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for accessor in data.get("accessors", []):
        if accessor.get("type") != "VEC3":
            continue
        if "coord" not in accessor.get("name", "").lower():
            continue
        mins.append(np.array(accessor["min"], dtype=float))
        maxs.append(np.array(accessor["max"], dtype=float))
    if not mins or not maxs:
        raise ValueError(f"Could not find coordinate bounds in mesh '{path}'.")
    return np.min(np.stack(mins), axis=0), np.max(np.stack(maxs), axis=0)


@dataclass(frozen=True)
class UrdfMechanismSpec:
    source_path: Path
    x_stage_prismatic_joint: str
    x_stage_screw_joint: str
    secondary_axis_joint: str
    pendulum_joints: tuple[str, str, str]
    rail_limit_m: float
    lead_m_per_rev: float
    carriage_width_m: float
    carriage_height_m: float
    pivot_offset_x_m: float
    pivot_offset_z_m: float
    lower_link_m: float
    middle_link_m: float
    upper_link_m: float

    @classmethod
    def from_v2_urdf(cls, path: Path) -> "UrdfMechanismSpec":
        if not path.exists():
            raise FileNotFoundError(f"Expected V2 URDF at '{path}'.")

        root = ET.parse(path).getroot()
        world_link_poses: dict[str, np.ndarray] = {"root": np.eye(4, dtype=float)}

        pending = list(root.findall("joint"))
        while pending:
            next_pending: list[ET.Element] = []
            progressed = False
            for joint in pending:
                parent = joint.find("parent")
                child = joint.find("child")
                origin = joint.find("origin")
                if parent is None or child is None or origin is None:
                    continue
                parent_link = parent.attrib["link"]
                child_link = child.attrib["link"]
                if parent_link not in world_link_poses:
                    next_pending.append(joint)
                    continue
                world_link_poses[child_link] = world_link_poses[parent_link] @ _origin_transform(origin)
                progressed = True
            if not progressed:
                break
            pending = next_pending

        def joint_world_xyz(name: str) -> np.ndarray:
            joint = root.find(f"joint[@name='{name}']")
            if joint is None:
                raise KeyError(f"Joint '{name}' not found in {path}.")
            parent = joint.find("parent")
            origin = joint.find("origin")
            if parent is None or origin is None:
                raise KeyError(f"Joint '{name}' is missing parent/origin data.")
            parent_link = parent.attrib["link"]
            if parent_link not in world_link_poses:
                raise KeyError(f"Unable to resolve world pose for parent '{parent_link}'.")
            return (world_link_poses[parent_link] @ _origin_transform(origin))[:3, 3]

        def link_inertial_length(name: str) -> float:
            link = root.find(f"link[@name='{name}']")
            if link is None:
                raise KeyError(f"Link '{name}' not found in {path}.")
            inertial_origin = link.find("inertial/origin")
            if inertial_origin is None:
                raise KeyError(f"Link '{name}' is missing inertial/origin in {path}.")
            return 2.0 * float(np.linalg.norm(_parse_vec3(inertial_origin.attrib["xyz"])))

        # V2 export uses the x-axis cylindrical mate for carriage travel and the
        # slower `<1>` motor revolute mate as the secondary driven axis.
        x_axis_world = joint_world_xyz("x_axis_1")
        pivot_world = joint_world_xyz("revolute_1_0")
        meshes_root = path.parents[1] / "meshes"
        part1_min, part1_max = _gltf_bounds(meshes_root / "Part 1.gltf")
        part1_size = part1_max - part1_min

        return cls(
            source_path=path,
            x_stage_prismatic_joint="x_axis_1",
            x_stage_screw_joint="x_axis_2",
            secondary_axis_joint="revolute_1_1",
            pendulum_joints=("revolute_1_0", "revolute_2_0", "revolute_3_0"),
            # The CAD-exported URDF uses placeholder +/-10000 limits. The local
            # physics/UI clamp to the physical rail range already used in the repo.
            rail_limit_m=0.14,
            lead_m_per_rev=8.0e-3,
            carriage_width_m=float(part1_size[0]),
            carriage_height_m=float(part1_size[2]),
            pivot_offset_x_m=float(pivot_world[0] - x_axis_world[0]),
            pivot_offset_z_m=float(pivot_world[2] - x_axis_world[2]),
            lower_link_m=link_inertial_length("part_5"),
            middle_link_m=link_inertial_length("part_6"),
            upper_link_m=link_inertial_length("part_7"),
        )


class MuJoCoPendulumModel:
    """MuJoCo kinematic model used to draw the 2D simulator view."""

    def __init__(self, mechanism: UrdfMechanismSpec) -> None:
        self.mechanism = mechanism
        self.model = mujoco.MjModel.from_xml_string(self._build_mjcf(mechanism))
        self.data = mujoco.MjData(self.model)

    @staticmethod
    def _build_mjcf(mechanism: UrdfMechanismSpec) -> str:
        limit = mechanism.rail_limit_m
        half_w = mechanism.carriage_width_m / 2.0
        half_h = mechanism.carriage_height_m / 2.0
        return f"""
<mujoco model="triple_inverted_pendulum_ui_v2">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4"/>
  <visual>
    <global offwidth="800" offheight="600"/>
    <rgba haze="0 0 0 1"/>
  </visual>
  <default>
    <geom contype="0" conaffinity="0" rgba="1 1 1 1"/>
    <site rgba="1 1 1 1" size="0.01"/>
  </default>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="{mechanism.x_stage_prismatic_joint}" type="slide" axis="1 0 0"
             range="{-limit} {limit}" limited="true"/>
      <geom name="cart_geom" type="box"
            size="{half_w:.6f} 0.045 {half_h:.6f}"
            pos="0 0 0"/>
      <site name="pivot" pos="{mechanism.pivot_offset_x_m:.6f} 0 {mechanism.pivot_offset_z_m:.6f}"/>
      <body name="link1" pos="{mechanism.pivot_offset_x_m:.6f} 0 {mechanism.pivot_offset_z_m:.6f}">
        <joint name="{mechanism.pendulum_joints[0]}" type="hinge" axis="0 1 0"/>
        <geom name="rod1" type="capsule"
              fromto="0 0 0 {mechanism.lower_link_m:.6f} 0 0"
              size="0.0115" rgba="1 1 1 1"/>
        <site name="tip1" pos="{mechanism.lower_link_m:.6f} 0 0"/>
        <body name="link2" pos="{mechanism.lower_link_m:.6f} 0 0">
          <joint name="{mechanism.pendulum_joints[1]}" type="hinge" axis="0 1 0"/>
          <geom name="rod2" type="capsule"
                fromto="0 0 0 {mechanism.middle_link_m:.6f} 0 0"
                size="0.0100" rgba="0.82 0.82 0.82 1"/>
          <site name="tip2" pos="{mechanism.middle_link_m:.6f} 0 0"/>
          <body name="link3" pos="{mechanism.middle_link_m:.6f} 0 0">
            <joint name="{mechanism.pendulum_joints[2]}" type="hinge" axis="0 1 0"/>
            <geom name="rod3" type="capsule"
                  fromto="0 0 0 {mechanism.upper_link_m:.6f} 0 0"
                  size="0.0090" rgba="0.62 0.62 0.62 1"/>
            <site name="tip3" pos="{mechanism.upper_link_m:.6f} 0 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

    def set_state_from_sim(self, state: np.ndarray) -> None:
        theta1 = float(state[1])
        theta2 = float(state[2])
        theta3 = float(state[3])

        self.data.qpos[:] = [
            float(np.clip(state[0], -self.mechanism.rail_limit_m, self.mechanism.rail_limit_m)),
            theta1 - np.pi / 2.0,
            theta2 - theta1,
            theta3 - theta2,
        ]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def points_2d(self) -> dict[str, tuple[float, float]]:
        return {
            "pivot": self._site_xz("pivot"),
            "tip1": self._site_xz("tip1"),
            "tip2": self._site_xz("tip2"),
            "tip3": self._site_xz("tip3"),
            "cart": (
                float(self.data.body("cart").xpos[0]),
                float(self.data.body("cart").xpos[2]),
            ),
        }

    def _site_xz(self, name: str) -> tuple[float, float]:
        pos = self.data.site(name).xpos
        return float(pos[0]), float(pos[2])


class TriplePendulumUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Triple Inverted Pendulum")
        self.configure(bg=BG)
        self.geometry("1440x900")
        self.minsize(1100, 700)

        self.mechanism = UrdfMechanismSpec.from_v2_urdf(V2_URDF_PATH)
        self.result: sim.SimulationResult | None = None
        self.viewer_model: MuJoCoPendulumModel | None = None
        self.frame_index = 0
        self.playing = False
        self.play_job: str | None = None

        self.motor_deg_var = tk.StringVar(value="+0.0°")
        self._build_ui()
        self.run_demo()

    def _build_ui(self) -> None:
        root = tk.Frame(self, bg=BG)
        root.pack(fill=tk.BOTH, expand=True)
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _e: self._draw_current_frame())

        panel = tk.Frame(root, bg=PANEL, width=230)
        panel.grid(row=0, column=1, sticky="ns")
        panel.grid_propagate(False)

        ttk.Style().theme_use("clam")
        style = ttk.Style()
        style.configure("Dark.TButton", font=("Segoe UI", 11, "bold"), padding=8)

        tk.Label(
            panel,
            text="MOTOR DEG",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w", padx=22, pady=(28, 6))
        tk.Label(
            panel,
            textvariable=self.motor_deg_var,
            bg=PANEL,
            fg=TEXT,
            font=("Consolas", 28, "bold"),
        ).pack(anchor="w", padx=22)

        controls = tk.Frame(panel, bg=PANEL)
        controls.pack(fill=tk.X, padx=18, pady=(28, 0))
        ttk.Button(controls, text="Replay", style="Dark.TButton", command=self.run_demo).pack(
            fill=tk.X, pady=6
        )
        self.play_button = ttk.Button(
            controls, text="Pause", style="Dark.TButton", command=self.toggle_play
        )
        self.play_button.pack(fill=tk.X, pady=6)

        tk.Label(
            panel,
            text="V2 URDF geometry\nx_axis_1 / x_axis_2\nsecondary: revolute_1_1",
            bg=PANEL,
            fg=MUTED,
            justify="left",
            font=("Segoe UI", 10),
        ).pack(anchor="w", padx=22, pady=(28, 0))

    def run_demo(self) -> None:
        self._cancel_playback()

        params, motor, controller, initial_state, use_motor_model = sim.config_v2_dual_balance()
        physics = sim.TriplePendulumPhysics(params)
        simulation = sim.Simulation(physics, controller, motor)
        self.result = simulation.run(
            t_span=(0.0, 5.0),
            initial_state=initial_state,
            dt_output=0.002,
            use_motor_model=use_motor_model,
        )
        self.viewer_model = MuJoCoPendulumModel(self.mechanism)
        self.frame_index = 0
        self.playing = True
        self.play_button.configure(text="Pause")
        self._draw_current_frame()
        self._play_step()

    def toggle_play(self) -> None:
        if self.result is None:
            return
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._play_step()
        else:
            self._cancel_playback()

    def _cancel_playback(self) -> None:
        self.playing = False
        if self.play_job is not None:
            self.after_cancel(self.play_job)
            self.play_job = None

    def _play_step(self) -> None:
        if not self.playing or self.result is None:
            return

        self._draw_current_frame()
        self.frame_index += 4
        if self.frame_index >= len(self.result.t):
            self.frame_index = 0

        self.play_job = self.after(16, self._play_step)

    def _draw_current_frame(self) -> None:
        if self.result is None or self.viewer_model is None:
            return

        state = self.result.states[self.frame_index]
        self.viewer_model.set_state_from_sim(state)
        points = self.viewer_model.points_2d()
        width = max(self.canvas.winfo_width(), 2)
        height = max(self.canvas.winfo_height(), 2)

        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, width, height, fill=BG, outline=BG)

        x_extent = self.mechanism.rail_limit_m + 0.12
        max_z = (
            self.mechanism.pivot_offset_z_m
            + self.mechanism.lower_link_m
            + self.mechanism.middle_link_m
            + self.mechanism.upper_link_m
            + 0.08
        )
        min_z = -(self.mechanism.carriage_height_m / 2.0 + 0.08)
        scale_x = width / (2.0 * x_extent)
        scale_y = height / (max_z - min_z)
        scale = min(scale_x, scale_y)
        origin_x = width * 0.50
        origin_y = height * 0.72 - min_z * scale

        def project(x: float, z: float) -> tuple[float, float]:
            return origin_x + x * scale, origin_y - z * scale

        rail_left = project(-self.mechanism.rail_limit_m, 0.0)
        rail_right = project(self.mechanism.rail_limit_m, 0.0)
        self.canvas.create_line(
            *rail_left,
            *rail_right,
            fill=TRACK,
            width=8,
            capstyle=tk.ROUND,
        )

        stop_height = self.mechanism.carriage_height_m * 0.85
        for x_stop in (-self.mechanism.rail_limit_m, self.mechanism.rail_limit_m):
            x0, y0 = project(x_stop, -0.02)
            x1, y1 = project(x_stop, stop_height)
            self.canvas.create_line(x0, y0, x1, y1, fill="#303030", width=4)

        cart_x, cart_z = points["cart"]
        pivot = project(*points["pivot"])
        tip1 = project(*points["tip1"])
        tip2 = project(*points["tip2"])
        tip3 = project(*points["tip3"])

        cart_center = project(cart_x, cart_z)
        cart_w = self.mechanism.carriage_width_m * scale
        cart_h = self.mechanism.carriage_height_m * scale
        self.canvas.create_rectangle(
            cart_center[0] - cart_w / 2.0,
            cart_center[1] - cart_h / 2.0,
            cart_center[0] + cart_w / 2.0,
            cart_center[1] + cart_h / 2.0,
            fill=CART,
            outline="",
        )

        self.canvas.create_line(*pivot, *tip1, fill=LINK_1, width=8, capstyle=tk.ROUND)
        self.canvas.create_line(*tip1, *tip2, fill=LINK_2, width=7, capstyle=tk.ROUND)
        self.canvas.create_line(*tip2, *tip3, fill=LINK_3, width=6, capstyle=tk.ROUND)

        for px, py, r in (
            (pivot[0], pivot[1], 6),
            (tip1[0], tip1[1], 5),
            (tip2[0], tip2[1], 5),
            (tip3[0], tip3[1], 5),
        ):
            self.canvas.create_oval(px - r, py - r, px + r, py + r, fill=JOINT, outline="")

        motor_deg = self._motor_deg_from_cart_x(float(state[0]))
        self.motor_deg_var.set(f"{motor_deg:+.1f}°")

    def _motor_deg_from_cart_x(self, cart_x_m: float) -> float:
        return cart_x_m / self.mechanism.lead_m_per_rev * 360.0


if __name__ == "__main__":
    app = TriplePendulumUI()
    app.mainloop()
