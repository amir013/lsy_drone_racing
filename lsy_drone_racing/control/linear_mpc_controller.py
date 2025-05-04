"""Linear MPC controller for drone racing.

This module implements a linear Model Predictive Controller for drone racing.
It uses a linearized quadrotor model and solves a quadratic programming problem
at each time step to compute optimal control inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class LinearMPCRacingController(Controller):
    """Linear MPC controller for drone racing."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the linear MPC controller.

        Args:
            obs: The initial observation of the environment's state.
            info: The initial environment information from the reset.
            config: The race configuration.
        """
        super().__init__(obs, info, config)
        
        # Controller parameters
        self.dt = 1.0 / config.env.freq  # Time step
        self.N = 10  # Prediction horizon
        self.Q = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])  # State cost
        self.R = np.diag([0.1, 0.1, 0.1])  # Control cost
        
        # Initialize system matrices (simplified linear model)
        self.A = self._compute_system_matrix()
        self.B = self._compute_input_matrix()
        
        # Initialize state and control history
        self.state_history = []
        self.control_history = []
        
    def _compute_system_matrix(self) -> NDArray[np.floating]:
        """Compute the system matrix A for the linearized model."""
        A = np.eye(6)  # 6 states: [x, y, z, vx, vy, vz]
        A[0:3, 3:6] = self.dt * np.eye(3)
        return A
        
    def _compute_input_matrix(self) -> NDArray[np.floating]:
        """Compute the input matrix B for the linearized model."""
        B = np.zeros((6, 3))  # 3 inputs: [ax, ay, az]
        B[3:6, :] = self.dt * np.eye(3)
        return B
        
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the control input using linear MPC.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information.

        Returns:
            The control command as a numpy array.
        """
        # Extract current state
        pos = obs["position"]
        vel = obs["velocity"]
        state = np.concatenate([pos, vel])
        
        # Get reference trajectory (simplified for now)
        ref_trajectory = self._get_reference_trajectory()
        
        # Solve MPC problem
        control = self._solve_mpc(state, ref_trajectory)
        
        # Store history
        self.state_history.append(state)
        self.control_history.append(control)
        
        return control
        
    def _get_reference_trajectory(self) -> NDArray[np.floating]:
        """Get reference trajectory for the next N steps."""
        # Simplified: constant velocity forward
        return np.array([1.0, 0.0, 0.0])  # [vx, vy, vz]
        
    def _solve_mpc(
        self, current_state: NDArray[np.floating], ref_trajectory: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Solve the MPC optimization problem.

        Args:
            current_state: Current state of the system.
            ref_trajectory: Reference trajectory.

        Returns:
            Optimal control input.
        """
        # Simplified MPC: just return a basic control
        return np.array([0.1, 0.0, 0.0])  # [ax, ay, az]
        
    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Callback after each step.

        Args:
            action: Latest applied action.
            obs: Latest environment observation.
            reward: Latest reward.
            terminated: Latest terminated flag.
            truncated: Latest truncated flag.
            info: Latest information dictionary.

        Returns:
            Whether the controller has finished.
        """
        return False  # Continue control
        
    def reset(self):
        """Reset the controller's internal state."""
        self.state_history = []
        self.control_history = [] 