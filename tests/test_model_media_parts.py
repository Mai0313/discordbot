"""Guards the project-wide rule that model media is uploaded, never handed over as a URL.

A remote http(s) URL in a media part looks like it works and mostly does, which is why this
is a lint rather than a comment: the LiteLLM proxy rewrites any http-bearing `file_id` /
`file_url` into base64 `inline_data`, charging the media against the request body and
swallowing a failed fetch (`except Exception: pass`), while the native Interactions answer
path has no proxy at all and only resolves Files API uris and YouTube links. Uploading via
`gen_reply/files_api.py` is the one shape both accept.

Data URIs are exempt: the bytes are already in hand, so nothing is fetched.

The scan is AST over every module under `src/`, so no cog has to be importable for the rule to
hold. It matches a part by the constructor's bare name (attribute-qualified or not) and reads
only the keyword it was built with, and the data-URI exemption is syntactic: an `image_url` is
cleared solely by a direct call to one of `_DATA_URI_BUILDERS`, so a data URI bound to a local
first is reported rather than slipped past.
"""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"

# Helpers that turn bytes already in hand into a `data:` URI. An `image_url` built by one of
# these is inlined on purpose (the non-Gemini renderer and the generated-media reply paths).
_DATA_URI_BUILDERS = frozenset({"convert_base64_to_data_uri", "_data_uri"})

_MEDIA_PART_CALLS = frozenset({"ResponseInputImageParam", "ResponseInputFileParam"})


def _called_name(node: ast.expr) -> str:
    """Returns the bare name of a call target, or an empty string for anything else.

    An attribute call yields its final attribute, so a module-qualified
    `mod.ResponseInputFileParam(...)` and a directly imported one resolve alike.
    """
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
    return ""


def _offending_arguments(call: ast.Call) -> list[str]:
    """Returns the media-source keywords of a part that are not a local data URI.

    `file_url` has no exemption at all, because `ResponseInputFileParam` carries in-hand bytes
    in `file_data` instead, so a `file_url` is remote by definition. `image_url` is the one
    field that also takes a `data:` URI, so it is cleared only when a `_DATA_URI_BUILDERS` call
    is what built it.
    """
    offenders: list[str] = []
    for keyword in call.keywords:
        if keyword.arg not in ("image_url", "file_url"):
            continue
        if keyword.arg == "file_url":
            offenders.append("file_url")
            continue
        if _called_name(keyword.value) not in _DATA_URI_BUILDERS:
            offenders.append("image_url")
    return offenders


def test_no_media_part_is_built_from_a_remote_url() -> None:
    """No media part in src/ carries a remote URL; media reaches the model via the Files API."""
    findings: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(source=path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _called_name(node) not in _MEDIA_PART_CALLS:
                continue
            findings.extend(
                f"{path.relative_to(_SRC)}:{node.lineno} passes {argument}"
                for argument in _offending_arguments(node)
            )
    assert findings == [], (
        "media parts must reference an uploaded Files API uri via file_id, not a remote URL "
        f"(see the module docstring): {findings}"
    )
