"""Smoke test — verifies the project structure and test infrastructure work."""

import ribo_switch


def test_package_imports():
    """The ribo_switch package should be importable."""
    assert ribo_switch.__version__ == "0.1.0"


def test_python_version():
    """Ensure we're running Python 3.11+, required for union type syntax."""
    import sys
    assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version}"
