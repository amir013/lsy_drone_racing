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
from scipy.interpolate import CubicSpline

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
        
        # Initialize trajectory
        self.waypoints = self._initialize_waypoints()
        self.trajectory = self._generate_trajectory()
        
    def _initialize_waypoints(self) -> NDArray[np.floating]:
        """Initialize waypoints for the race track."""
        # Example waypoints - you can modify these based on the track
        return np.array([
            [0.0, 0.0, 1.0],  # Start
            [1.0, 0.0, 1.0],  # First gate
            [2.0, 1.0, 1.0],  # Second gate
            [1.0, 2.0, 1.0],  # Third gate
            [0.0, 1.0, 1.0],  # Fourth gate
        ])
        
    def _generate_trajectory(self) -> CubicSpline:
        """Generate a smooth trajectory through waypoints."""
        t = np.linspace(0, 1, len(self.waypoints))
        return CubicSpline(t, self.waypoints)
        
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
        
        # Get reference trajectory
        ref_trajectory = self._get_reference_trajectory(state)
        
        # Solve MPC problem
        control = self._solve_mpc(state, ref_trajectory)
        
        # Store history
        self.state_history.append(state)
        self.control_history.append(control)
        
        return control
        
    def _get_reference_trajectory(self, current_state: NDArray[np.floating]) -> NDArray[np.floating]:
        """Get reference trajectory for the next N steps."""
        # Get current position
        current_pos = current_state[:3]
        
        # Find closest point on trajectory
        t_values = np.linspace(0, 1, 100)
        trajectory_points = self.trajectory(t_values)
        distances = np.linalg.norm(trajectory_points - current_pos, axis=1)
        closest_idx = np.argmin(distances)
        
        # Get next N points on trajectory
        t_next = np.linspace(t_values[closest_idx], 1, self.N)
        ref_points = self.trajectory(t_next)
        
        return ref_points
        
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
        def cost_function(u):
            # Reshape control sequence
            u = u.reshape((self.N, 3))
            
            # Simulate system forward
            x = current_state.copy()
            cost = 0.0
            
            for i in range(self.N):
                # State cost
                error = x[:3] - ref_trajectory[i]
                cost += error.T @ self.Q[:3, :3] @ error
                
                # Control cost
                cost += u[i].T @ self.R @ u[i]
                
                # Update state
                x = self.A @ x + self.B @ u[i]
            
            return cost
        
        # Initial guess for control sequence
        u0 = np.zeros(self.N * 3)
        
        # Bounds on control inputs
        bounds = [(-2.0, 2.0) for _ in range(self.N * 3)]
        
        # Solve optimization problem
        result = minimize(
            cost_function,
            u0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100}
        )
        
        # Return first control input
        return result.x[:3]
        
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
        self.trajectory = self._generate_trajectory() 