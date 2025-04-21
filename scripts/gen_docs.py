# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "anyio",
#     "fire",
#     "notebook",
#     "pydantic",
#     "rich",
# ]
# ///
import ast
import shutil
from typing import Literal
from pathlib import Path
from functools import cached_property
import itertools

import anyio
import nbformat
from pydantic import Field, BaseModel, ConfigDict, computed_field
from nbconvert import MarkdownExporter, TemplateExporter
from rich.console import Console
from rich.progress import Progress

console = Console()


class DocsGenerator(BaseModel):
    """DocsGenerator is a class that generates documentation for Python files or classes within a specified source directory.

    Attributes:
        source (str): The source directory or file path.
        output (str): The output directory path.
        exclude (str): Comma-separated list of folders or files to exclude.
        mode (Literal["file", "class"]): Mode of documentation generation, either by file or class.

    Methods:
        gen_docs() -> None:
            Generates documentation by file or class.

        __call__() -> None:
            Asynchronously calls the gen_docs method.

    Using CLI:
        ```bash
        python ./scripts/gen_docs.py --source ./src --output ./docs/Reference --exclude .venv gen_docs
        ```

    Using Rye:
        ```bash
        uv run python ./scripts/gen_docs.py
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    source_path: Path = Field(
        ...,
        title="The Source File Path or Folder Path",
        description="This field can be a file path or folder path, if it is a folder path, it will automatically search for python and ipynb files.",
        examples=["./src"],
        alias="source",
        frozen=True,
    )
    output_path: Path = Field(
        ...,
        title="The Output Path",
        description="The output path for the generated documentation.",
        examples=["./docs/Reference"],
        alias="output",
        frozen=True,
    )
    exclude: str = Field(
        default=".venv",
        description="Exclude the folder or file, it should be separated by comma.",
        examples=[".venv,.git,.idea"],
    )
    mode: Literal["file", "class"] = Field(
        default="class",
        title="The Document Style",
        description="Generate docs by file or class.",
        examples=["file", "class"],
    )

    @computed_field
    @cached_property
    def source_files(self) -> list[Path]:
        """Computed property that returns the source path as a Path object.

        Returns:
            Path: The source path.
        """
        if self.source_path.is_dir():
            if self.output_path.exists():
                shutil.rmtree(self.output_path.absolute())
                self.output_path.mkdir(parents=True, exist_ok=True)
            need_to_exclude = [*self.exclude.split(","), ".venv", "__init__.py"]
            need_to_exclude = list(set(need_to_exclude))
            python_files = self.source_path.rglob("*.py")
            ipynb_files = self.source_path.rglob("*.ipynb")
            files = itertools.chain(python_files, ipynb_files)
            all_files = [
                file for file in files if not any(f in file.parts for f in need_to_exclude)
            ]
        elif self.source_path.is_file():
            all_files = [self.source_path]
        else:
            raise ValueError("Invalid source path")
        return all_files

    async def __prepare_docs_path(self, file: Path) -> Path:
        # 因為多層結構的資料夾 我們希望他可以依然放在對應的資料夾內
        try:
            related_path = file.parent.relative_to(self.source_path)
        except ValueError:
            related_path_str = file.parent.as_posix().replace(self.source_path.as_posix(), "")
            related_path = Path(related_path_str)
        filename = file.with_suffix(".md").name
        docs_path = Path(f"{self.output_path}/{related_path}/{filename}")
        docs_path.parent.mkdir(parents=True, exist_ok=True)
        docs_path.unlink(missing_ok=True)
        return docs_path

    async def __gen_python_docs(self, file: Path) -> str:
        docs_path = await self.__prepare_docs_path(file=file)
        if self.mode == "file":
            note_content = f"::: {file.as_posix().removesuffix('.py').replace('/', '.')}\n"
        elif self.mode == "class":
            async with await anyio.open_file(file, encoding="utf-8") as f:
                contents = await f.read()
                tree = ast.parse(source=contents, filename=file)

            note_content = ""
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name.startswith("_"):
                        continue
                    note_content += f"::: {file.as_posix().removesuffix('.py').replace('/', '.')}.{node.name}\n"
        else:
            raise ValueError("Invalid mode")
        if not note_content:
            note_content = f"::: {file.as_posix().removesuffix('.py').replace('/', '.')}\n"
        docs_path.write_text(data=note_content, encoding="utf-8")
        return docs_path.as_posix()

    async def __gen_notebook_docs(self, file: Path) -> str:
        docs_path = await self.__prepare_docs_path(file=file)
        async with await anyio.open_file(file, encoding="utf-8") as f:
            nb = nbformat.reads(await f.read(), as_version=4)

        markdown_exporter = MarkdownExporter(template_name="markdown")
        if not isinstance(markdown_exporter, TemplateExporter):
            raise TypeError("TemplateExporter is not a valid type")
        markdown_output, _ = markdown_exporter.from_notebook_node(nb)

        async with await anyio.open_file(docs_path, "w", encoding="utf-8") as f:
            await f.write(markdown_output)
        return docs_path.as_posix()

    async def gen_docs(self) -> None:
        with Progress() as progress:
            task = progress.add_task("[green]Generating docs...")

            progress.update(
                task_id=task, description="[cyan]Files Found...", total=len(self.source_files)
            )

            for source_file in self.source_files:
                progress.update(
                    task_id=task,
                    advance=1,
                    description=f"[cyan]Processing {source_file.as_posix()}...",
                    refresh=True,
                )
                if source_file.suffix == ".ipynb":
                    await self.__gen_notebook_docs(file=source_file)
                elif source_file.suffix == ".py":
                    await self.__gen_python_docs(file=source_file)
                else:
                    console.log(f"Unsupported file type: {source_file.suffix}")

    async def __call__(self) -> None:
        """Asynchronously calls the gen_docs method.

        This method is intended to be used as a callable object, which when called, will execute the gen_docs method asynchronously.

        Examples:
            >>> import asyncio
            >>> pair_list = {"./src": "./docs/Reference"}
            >>> for key, value in pair_list.items():
            ...     docs_generator = DocsGenerator(source=key, output=value, exclude=".venv", mode="class")
            ...     asyncio.run(docs_generator())
        """
        await self.gen_docs()


if __name__ == "__main__":
    import fire

    fire.Fire(DocsGenerator)
