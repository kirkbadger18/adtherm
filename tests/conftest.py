import sys
from pathlib import Path

# Make the package importable when tests run without an editable install
# (`pip install -e .` also puts the repo root on the path).
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
