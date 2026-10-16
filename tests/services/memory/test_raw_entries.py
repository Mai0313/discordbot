"""Tests for rendering observations into raw entries."""

from discordbot.services.memory.raw_entries import fields_of, render_memory_observations

from tests.helpers.user_memory import _observation


def test_render_memory_observations_stamps_source_and_sharing() -> None:
    rendered = render_memory_observations(
        observations=(_observation(summary="喜歡簡短", sharing="source_only"),), source="guild 123"
    )
    fields = fields_of(block=rendered)
    assert fields["source"] == "guild 123"
    assert fields["sharing"] == "source_only"


def test_render_memory_observations_without_source_omits_source_and_sharing() -> None:
    # The server flavor renders neither field.
    rendered = render_memory_observations(
        observations=(_observation(summary="喜歡簡短"),), source=None
    )
    assert "- source:" not in rendered
    assert "- sharing:" not in rendered
