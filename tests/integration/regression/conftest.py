"""
Fixtures para tests de regresión.
"""

import pytest
import json
from pathlib import Path


@pytest.fixture
def golden_outputs_dir():
    """Directorio de golden outputs."""
    return Path(__file__).parent.parent.parent / "data" / "golden"


@pytest.fixture
def load_golden_output(golden_outputs_dir):
    """Función para cargar golden outputs."""
    def _load(filename):
        path = golden_outputs_dir / filename
        with open(path) as f:
            return json.load(f)
    return _load

