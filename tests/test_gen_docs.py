"""Tests for the documentation generator script."""

from pathlib import Path

from scripts.gen_docs import DocsGenerator


def _previous_run(tmp_path: Path) -> DocsGenerator:
    """A generator over one source module whose output still holds a page from an earlier run."""
    module = tmp_path / "src" / "pkg" / "mod.py"
    module.parent.mkdir(parents=True)
    module.write_text(data="class Thing:\n    pass\n", encoding="utf-8")
    stale = tmp_path / "out" / "removed.md"
    stale.parent.mkdir()
    stale.write_text(data="stale\n", encoding="utf-8")
    return DocsGenerator(source=tmp_path / "src", output=tmp_path / "out")


def test_inspecting_the_generator_leaves_the_output_alone(tmp_path: Path) -> None:
    """Dumping the model lists the sources without deleting a previous run's pages."""
    generator = _previous_run(tmp_path=tmp_path)

    dumped = generator.model_dump()

    assert dumped["source_files"] == [tmp_path / "src" / "pkg" / "mod.py"]
    assert (tmp_path / "out" / "removed.md").exists()


async def test_generating_drops_pages_left_by_a_previous_run(tmp_path: Path) -> None:
    """A run over a source directory replaces the output, so a removed module's page is gone."""
    generator = _previous_run(tmp_path=tmp_path)

    await generator.gen_docs()

    assert not (tmp_path / "out" / "removed.md").exists()
    assert (tmp_path / "out" / "pkg" / "mod.md").exists()
