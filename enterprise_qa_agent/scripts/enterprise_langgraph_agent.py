from __future__ import annotations

import sys
import os
from pathlib import Path


# Edit this when you want to run a different atomic question from code.
TARGET_ATOMIC_ID = "fin_a_009_B"

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
os.environ.setdefault("ATOMIC_ID", TARGET_ATOMIC_ID)

from enterprise_qa_agent.src.legacy_agent import main


if __name__ == "__main__":
    main()
