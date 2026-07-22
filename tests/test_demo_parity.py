import pathlib
import shutil
import subprocess

import pytest

DEMO = pathlib.Path(__file__).resolve().parent.parent / "demo"


@pytest.mark.skipif(shutil.which("node") is None or not (DEMO / "model.json").exists(),
                    reason="needs node and an exported demo/model.json")
def test_browser_forward_matches_pytorch():
    result = subprocess.run(["node", str(DEMO / "parity.mjs")], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
