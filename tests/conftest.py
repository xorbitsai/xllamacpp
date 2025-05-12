from pathlib import Path

ROOT = Path(__file__).parent.parent


import pytest


@pytest.fixture(scope="module")
def model_path():
    return str(ROOT / "models")
