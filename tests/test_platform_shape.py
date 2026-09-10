"""Pins the entry point every platform downloader answers on.

`tests/test_link_source_shape.py` holds the three conversation MODELS to one surface and says
nothing about what produces them — which is how five downloaders answering the same question ended
up disagreeing on how to ask it. Two allowed a positional `url` where three were keyword-only, and
two spelled the download half `download` while one called it `parse`. None of that was visible to
any test.

So the classes here are DISCOVERED, the same way and for the same reason: every `*Downloader` under
`discordbot.services.platforms` is swept up, and each guard covers a new platform without being
told about it. `test_the_sweep_finds_every_downloader` is the deliberate exception — it names the
five, so a sixth fails until somebody writes the name down and reads this file.

`PlatformDownloader` is excluded BY IDENTITY rather than by the module it lives in. Sweeping it
would check the base against itself — its `parse_metadata` is the `NotImplementedError` stub every
assertion below exists to say a platform replaced, and its declared return is the bare `BaseModel`
a platform must narrow. Skipping `base.py` wholesale would do the same job today and stop doing it
the moment that file gains a second class, so the exclusion names the one thing it means, and
`test_the_sweep_finds_every_downloader` asserts what was excluded as well as what was kept.
"""

from typing import cast
from inspect import Parameter, signature
from pkgutil import walk_packages
from importlib import import_module

import pytest
from pydantic import BaseModel

# Namespace import: the package object itself is the input, for its __path__.
import discordbot.services.platforms
from discordbot.services.platforms.base import PlatformDownloader

# Every platform reading a link today. A sixth has to be written in here, which is the step that
# makes someone read the rules above.
_EXPECTED = frozenset({
    "ThreadsDownloader",
    "FacebookDownloader",
    "InstagramDownloader",
    "DouyinDownloader",
    "VideoDownloader",
})


def _all_downloader_classes() -> dict[str, type[BaseModel]]:
    """Every `*Downloader` under `discordbot.services.platforms`, the base included.

    `walk_packages` rather than `iter_modules`, so a platform written as a subpackage is still
    found. With the top level alone its classes reach neither this sweep nor the tripwire, and a
    name nothing collected cannot make a set-equality assertion fail — it passes by matching an
    expectation that also does not list it.
    """
    found: dict[str, type[BaseModel]] = {}
    for module in walk_packages(
        path=discordbot.services.platforms.__path__, prefix="discordbot.services.platforms."
    ):
        imported = import_module(name=module.name)
        for name, value in vars(imported).items():
            if not name.endswith("Downloader") or not isinstance(value, type):
                continue
            if issubclass(value, BaseModel) and value.__module__ == imported.__name__:
                found[name] = value
    return found


def _downloader_classes() -> dict[str, type[PlatformDownloader]]:
    """The platform downloaders: everything above except the contract itself."""
    return {
        name: cast("type[PlatformDownloader]", value)
        for name, value in _all_downloader_classes().items()
        if value is not PlatformDownloader
    }


def test_the_sweep_finds_every_downloader() -> None:
    """A guard over a set it failed to collect would pass by finding nothing.

    Both halves are named, because the sweep can lose a platform two ways: by not reaching its
    module, and by being excluded alongside the contract. What the exclusion removed is therefore
    asserted too — `PlatformDownloader` and nothing else.
    """
    assert set(_downloader_classes()) == set(_EXPECTED)
    assert set(_all_downloader_classes()) - set(_EXPECTED) == {"PlatformDownloader"}


@pytest.mark.parametrize("name", sorted(_downloader_classes()))
def test_every_downloader_answers_the_shared_contract(name: str) -> None:
    """Inheriting the base is what makes `parse_metadata` a contract rather than a coincidence."""
    assert issubclass(_downloader_classes()[name], PlatformDownloader)


@pytest.mark.parametrize("name", sorted(_downloader_classes()))
def test_every_downloader_declares_its_own_parse_metadata(name: str) -> None:
    """A platform that inherited the stub would raise on its first real call instead of at import."""
    cls = _downloader_classes()[name]

    assert "parse_metadata" in vars(cls)


@pytest.mark.parametrize("name", sorted(_downloader_classes()))
def test_parse_metadata_takes_a_keyword_only_url(name: str) -> None:
    """One spelling at every call site, so a platform swap is not also a signature change.

    Keyword-only rather than merely keyword-able: every caller already writes `url=`, so a
    positional-or-keyword declaration is an invitation nothing has taken up yet. It had been taken
    up by two test doubles, which is exactly how a convention stops being one.
    """
    # Read off the class, so the signature is the unbound function's and carries `self`.
    parameters = signature(_downloader_classes()[name].parse_metadata).parameters
    arguments = [parameter for parameter in parameters if parameter != "self"]

    assert arguments == ["url"], f"{name}.parse_metadata takes more than a url"
    assert parameters["url"].kind is Parameter.KEYWORD_ONLY
    assert parameters["url"].annotation is str


@pytest.mark.parametrize("name", sorted(_downloader_classes()))
def test_parse_metadata_returns_a_model_of_its_own_module(name: str) -> None:
    """The return narrows to this platform's `<Platform>Conversation` or `<Platform>Metadata`.

    Declared in the same module, which is what stops a platform answering with another's model: the
    shape of a Douyin post is Douyin's to describe, and a shared return type would have to be the
    union of everything or the intersection of nothing.
    """
    cls = _downloader_classes()[name]
    returns = signature(cls.parse_metadata).return_annotation

    assert isinstance(returns, type), f"{name}.parse_metadata returns {returns!r}, not a class"
    assert issubclass(returns, BaseModel), f"{name}.parse_metadata does not return a model"
    assert returns is not BaseModel, f"{name}.parse_metadata did not narrow the base's return"
    assert returns.__module__ == cls.__module__, (
        f"{name}.parse_metadata returns {returns.__name__}, declared in another module"
    )


@pytest.mark.parametrize("name", sorted(_downloader_classes()))
def test_a_download_folder_is_required_when_it_exists(name: str) -> None:
    """A default would put scratch files somewhere nobody chose, and `data/` is one typo away.

    Only a platform that writes a file has the field at all: Facebook and Instagram hand image
    URLs to Discord and keep no state, so one instance serves every caller there.
    """
    field = _downloader_classes()[name].model_fields.get("output_folder")
    if field is None:
        return

    assert field.is_required(), f"{name}.output_folder has a default"


@pytest.mark.parametrize("name", sorted(_downloader_classes()))
def test_a_parse_half_takes_the_same_keyword_only_url(name: str) -> None:
    """`parse` is optional, but a platform that has one spells its url like `parse_metadata` does.

    Only Threads has it today. Three platforms write files, but their second methods have nothing
    a base could hold them to: Threads yields a conversation from a context manager, because what
    it cleans up hangs off that conversation, while Douyin and yt-dlp spell theirs `download` and
    return a `TemporaryDownload`, taking their own options alongside the url. That is why the base
    declares no second method at all — see its module docstring.
    """
    parse = getattr(_downloader_classes()[name], "parse", None)
    if parse is None:
        return

    url = signature(parse).parameters["url"]

    assert url.kind is Parameter.KEYWORD_ONLY
    assert url.annotation is str
