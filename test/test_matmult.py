import subprocess
import os

def test_matmult_runs_without_abort():
    # Get the absolute path to the repository root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run(
        [
            "python3", "-m", "src.codesign",
            "-b", "matmult",
            "--num_iters", "1",
            "--savedir", "test_logs"
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,  # Run from the top-level codesign module
        timeout=600  # 10 minutes, adjust as needed
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    assert result.returncode == 0, "Codesign aborted or failed (see stderr above)"