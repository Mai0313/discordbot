"""Tests for the documentation generator script."""

from pathlib import Path

import anyio
import pytest
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


async def test_a_module_with_classes_also_lists_its_public_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A class directive renders that class alone, so each public function gets its own."""
    package = tmp_path / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "mixed.py").write_text(
        data=(
            "class Thing:\n    def method(self):\n        pass\n\n\n"
            "def helper():\n    pass\n\n\n"
            "async def fetch():\n    pass\n\n\n"
            "def _private():\n    pass\n"
        ),
        encoding="utf-8",
    )
    (package / "functions.py").write_text(data="def helper():\n    pass\n", encoding="utf-8")
    monkeypatch.chdir(path=tmp_path)

    await DocsGenerator(source=Path("src"), output=Path("out")).gen_docs()

    assert await anyio.Path("out/pkg/mixed.md").read_text(encoding="utf-8") == (
        "::: src.pkg.mixed.Thing\n::: src.pkg.mixed.helper\n::: src.pkg.mixed.fetch\n"
    )
    assert await anyio.Path("out/pkg/functions.md").read_text(encoding="utf-8") == (
        "::: src.pkg.functions\n"
    )
