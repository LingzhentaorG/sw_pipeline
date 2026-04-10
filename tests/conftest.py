import pytest
from pathlib import Path


@pytest.fixture
def temp_storage_root(tmp_path: Path) -> Path:
    root = tmp_path / "storage"
    root.mkdir()
    return root
