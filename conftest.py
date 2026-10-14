# conftest.py — pytest configuration
# Adds project root to sys.path so all tests can import project modules
import sys
from pathlib import Path

# Ensure the project root is on the path for all tests
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
