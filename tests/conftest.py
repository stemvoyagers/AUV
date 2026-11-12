import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "simulator" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from auv_simulator.main import build_reference_simulator
from auv_simulator.trajectory import Trajectory


@pytest.fixture
def trajectory() -> Trajectory:
    return Trajectory()


@pytest.fixture
def reference_sim():
    return build_reference_simulator(seed=7)
