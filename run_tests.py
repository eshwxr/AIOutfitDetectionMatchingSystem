"""Script to run all tests"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run pytest tests"""
    test_dir = Path(__file__).parent / "tests"
    
    # Run pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"],
        cwd=Path(__file__).parent
    )
    
    return result.returncode == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

