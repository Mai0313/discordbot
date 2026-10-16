"""Tests for the Antigravity agent SDK development script."""

from importlib import import_module


def test_the_script_imports_on_the_locked_protobuf() -> None:
    """The SDK's generated protobuf code accepts the protobuf runtime the lock resolves."""
    import_module(name="scripts.antigravity_dev")
