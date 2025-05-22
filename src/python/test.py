# one time julia in the enviroment:
# export PATH="$HOME/.julia/environments/pyjuliapkg/pyjuliapkg/install/bin:$PATH"

from __future__ import annotations
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

_PROJECT_DIR = Path(__file__).resolve().parents[2]  # ~/GIV.jl repo root
_SRC_DIR = _PROJECT_DIR / "src"

# ---------------------------------------------------------------------------
# Julia environment setup
# ---------------------------------------------------------------------------
def _julia_cmd(test_file: Path) -> List[str]:
    """Build Julia command with environment setup"""
    return [
        "julia",
        "--project=" + str(_PROJECT_DIR),
        "-e", 
        f"""
        using Pkg
        Pkg.activate("{_PROJECT_DIR}")
        push!(LOAD_PATH, "{_SRC_DIR}")
        include("{test_file}")
        """
    ]

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
def run_test(test_file: str) -> Tuple[bool, str]:
    """Run a single test file with proper environment setup"""
    test_path = _PROJECT_DIR / "test" / test_file
    
    if not test_path.exists():
        return False, f"Test file not found: {test_path}"
    
    try:
        result = subprocess.run(
            _julia_cmd(test_path),
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code {e.returncode}\n"
        error_msg += f"STDOUT:\n{e.stdout}\n"
        error_msg += f"STDERR:\n{e.stderr}"
        return False, error_msg

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
TEST_FILES = [
    "test_formula.jl",
    "test_interface.jl",
    "test_estimates.jl",
    "test_with_simulation.jl",
    "benchmark_performance.jl"
]

def main():
    print(f"Running tests from: {_PROJECT_DIR}")
    print(f"Source directory: {_SRC_DIR}")
    
    passed = []
    failed = []
    
    for test_file in TEST_FILES:
        print(f"\n{'='*40}")
        print(f"Running {test_file}")
        success, output = run_test(test_file)
        
        if success:
            print(f"✅ {test_file} passed")
            passed.append(test_file)
        else:
            print(f"❌ {test_file} failed")
            failed.append(test_file)
        
        print(output)
    
    # Summary
    print(f"\nResults: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("Failed tests:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()