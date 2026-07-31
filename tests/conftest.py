from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).parents[1] / "unirl-reward-service"
sys.path.insert(0, str(SERVICE_ROOT))
