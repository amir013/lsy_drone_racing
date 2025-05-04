"""Control module for drone racing.

This module contains the base controller class that defines the interface for all controllers. Your
own controller goes in this module. It has to inherit from the base class and adhere to the same
function signatures.

To give you an idea of what you need to do, we also include some example implementations:

* :class:`~.Controller`: The abstract base class defining the interface for all controllers.
* :class:`TrajectoryController <lsy_drone_racing.control.trajectory_controller.TrajectoryController>`:
  A simple controller that follows a predefined trajectory.
* :class:`LinearMPCRacingController <lsy_drone_racing.control.linear_mpc_controller.LinearMPCRacingController>`:
  A linear MPC controller for drone racing.
"""

from lsy_drone_racing.control.controller import Controller
from lsy_drone_racing.control.trajectory_controller import TrajectoryController
from lsy_drone_racing.control.linear_mpc_controller import LinearMPCRacingController

__all__ = ["Controller", "TrajectoryController", "LinearMPCRacingController"]
