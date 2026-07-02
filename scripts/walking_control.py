"""Wheel-speed walking control for the wheeled biped.

Walking v1 keeps the current symmetric standing leg pose and commands the
wheel balance controller to track a ramped forward velocity.  In this model,
visual robot-forward corresponds to negative world-X motion.  Because the
existing wheel balance controller already contains the actuator-sign mapping,
positive ``forward_velocity`` maps to a positive balance-controller
``x_velocity_target``; the resulting base motion is toward world -X.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import mujoco
import numpy as np

from scripts.balance_control import (
    BalanceConfig,
    BalanceState,
    apply_balance_control,
    default_standing_config,
    standing_leg_targets,
)
from scripts.pd_control import JointControlMap


DEFAULT_FORWARD_VELOCITY = 0.25
DEFAULT_RAMP_TIME = 2.0
DEFAULT_WALKING_KV = 6.0
DEFAULT_WALKING_KX = 0.0


@dataclass(frozen=True)
class WalkingConfig:
    """User-facing wheel walking parameters.

    ``forward_velocity`` is positive in the robot's visual forward direction.
    The current MJCF convention makes this move the base toward world -X.
    """

    forward_velocity: float = DEFAULT_FORWARD_VELOCITY
    ramp_time: float = DEFAULT_RAMP_TIME
    pitch_target: float = 0.0
    pitch_rate_target: float = 0.0
    kp_pitch: float = 35.0
    kd_pitch: float = 6.0
    kv: float = DEFAULT_WALKING_KV
    leg_kp: float = 60.0
    leg_kd: float = 4.0


@dataclass(frozen=True)
class WalkingState:
    """Summary of one walking-control step."""

    forward_velocity_target: float
    balance_x_velocity_target: float
    forward_velocity: float
    forward_distance: float
    balance_state: BalanceState


def ramped_forward_velocity(config: WalkingConfig, time: float) -> float:
    """Return a smooth velocity command ramped from zero to target."""
    if config.ramp_time <= 0.0:
        return float(config.forward_velocity)
    alpha = float(np.clip(float(time) / float(config.ramp_time), 0.0, 1.0))
    return float(config.forward_velocity) * alpha


def balance_x_velocity_target(config: WalkingConfig, time: float) -> float:
    """Return the x-velocity command expected by ``compute_balance_control``.

    This is a controller command, not the resulting world-X velocity.  Positive
    values roll the current model toward world -X because wheel actuator signs
    are handled inside the balance controller.
    """
    return ramped_forward_velocity(config, time)


def walking_balance_config(config: WalkingConfig, time: float) -> BalanceConfig:
    """Build the underlying balance-controller config for a walking step."""
    base = default_standing_config()
    return replace(
        base,
        pitch_target=float(config.pitch_target),
        pitch_rate_target=float(config.pitch_rate_target),
        x_target=None,
        x_velocity_target=balance_x_velocity_target(config, time),
        kx=DEFAULT_WALKING_KX,
        kv=float(config.kv),
        kp_pitch=float(config.kp_pitch),
        kd_pitch=float(config.kd_pitch),
        leg_kp=float(config.leg_kp),
        leg_kd=float(config.leg_kd),
    )


def compute_forward_distance(data: mujoco.MjData, initial_x: float) -> float:
    """Return robot-forward distance traveled from the initial world-X position."""
    return -(float(data.qpos[0]) - float(initial_x))


def compute_forward_velocity(data: mujoco.MjData) -> float:
    """Return robot-forward velocity from MuJoCo world-X base velocity."""
    return -float(data.qvel[0])


def apply_walking_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_map: list[JointControlMap],
    *,
    config: WalkingConfig | None = None,
    leg_targets: Mapping[str, float] | None = None,
    initial_x: float = 0.0,
) -> tuple[np.ndarray, WalkingState]:
    """Apply one walking-control step to ``data.ctrl``."""
    config = config or WalkingConfig()
    targets = standing_leg_targets() if leg_targets is None else leg_targets
    balance_config = walking_balance_config(config, float(data.time))
    ctrl, balance_state = apply_balance_control(
        model,
        data,
        joint_map,
        balance_config,
        targets,
    )
    state = WalkingState(
        forward_velocity_target=ramped_forward_velocity(config, float(data.time)),
        balance_x_velocity_target=balance_config.x_velocity_target,
        forward_velocity=compute_forward_velocity(data),
        forward_distance=compute_forward_distance(data, initial_x),
        balance_state=balance_state,
    )
    return ctrl, state
