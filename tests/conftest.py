"""Pytest configuration and shared fixtures for ribo_switch tests."""

import pytest


@pytest.fixture
def adenine_on_structure() -> str:
    """Adenine riboswitch ON-state dot-bracket (approximate — verify against Rfam RF00167)."""
    return "...(((((((...((((((.........))))))........((((((.......))))))...)))))))..."


@pytest.fixture
def adenine_off_structure() -> str:
    """Adenine riboswitch OFF-state dot-bracket (approximate — verify against literature)."""
    return "...(((((((............((((((..........))))))((((((........))))))..)))))))..."


@pytest.fixture
def adenine_native_sequence() -> str:
    """Native adenine riboswitch sequence (verify against Rfam RF00167)."""
    return "CGCUUCAUAUAAUCCUAAUGAUAUGGUUUGGGAGUUUCUACCAAGAGCCUUAAACUCUUGAUUAUGAAGUG"
