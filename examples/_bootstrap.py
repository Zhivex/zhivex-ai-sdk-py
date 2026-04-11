from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_dotenv_if_available() -> bool:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    load_dotenv()
    return True
