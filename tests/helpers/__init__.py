"""Shared test helpers.

Structural assertion utilities and reusable test doubles live here so individual
test modules stop re-deriving brittle string checks and duplicating fake objects.
Modules in this package are import-safe and carry no doctests, so the suite's
``--doctest-modules`` collection stays clean.
"""
