"""Navigation estimators for the AUV simulator.

Design philosophy
-----------------
Keep the filter core tiny and numerical-stable; push all vehicle-specific
knowledge (what "position" means, what sensors measure) into separate
measurement-update functions following the experience pattern from ignav.

Error-State Kalman Filter (ESKF)
    * Propagates the **nominal** state (position, velocity, quaternion,
      accel bias, gyro bias) forward in time via IMU readings.
    * Tracks error state as a small perturbation δx on top of the nominal.
    * On each sensor reading we run a Joseph-form update to correct the
      nominal state.

This matches what the C++ side should implement — identical math.
"""

from .eskf import ESKF, ErrorState, NominalState

__all__ = ["ESKF", "ErrorState", "NominalState"]
