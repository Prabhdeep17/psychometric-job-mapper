import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest

@pytest.fixture(scope="session")
def data_scientists():
    from src.analysis import analyze
    return analyze(onet_soc_code="15-2051.00")
