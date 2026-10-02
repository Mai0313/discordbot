"""Guards the project-wide rule that model media is uploaded, never handed over as a URL.

A remote http(s) URL in a media part looks like it works and mostly does, which is why this
is a lint rather than a comment: the LiteLLM proxy rewrites any http-bearing `file_id` /
`file_url` other than a Gemini Files API uri into base64 `inline_data`, charging the media
against the request body and swallowing a failed fetch (`except Exception: pass`), while the
native Interactions answer path has no proxy at all and is only ever handed Files API uris and
YouTube links; whether it resolves an arbitrary remote URL was never validly measured (#346). Uploading via `gen_reply/files_api.py` is the one shape both accept.

Data URIs are exempt: the bytes are already in hand, so nothing is fetched.

What this cannot catch: a `file_id` is never checked, because it carries the Files API uri by
design, so a remote URL handed to one passes; only `file_url` and a non-data-URI `image_url` are
flagged, and only where the part is built in place as a TypedDict call or a dict literal.
"""

import ast

from tests.helpers.source_tree import PACKAGE, called_name, python_modules

# The helper that turns bytes already in hand into a `data:` URI. An `image_url` built by it is
# inlined on purpose (the non-Gemini renderer and the generated-media reply paths).
_DATA_URI_BUILDER = "to_data_uri"

_MEDIA_PART_CALLS = frozenset({"ResponseInputImageParam", "ResponseInputFileParam"})
_MEDIA_PART_TYPES = frozenset({"input_image", "input_file"})


def _media_part_fields(node: ast.expr) -> list[tuple[str, ast.expr]] | None:
    """Returns the fields a media part is built with, or None when `node` builds none.

    A part reaches the model the same way whether it is built through its TypedDict or written
    as a dict literal whose `type` names a media part, so both shapes are read.
    """
    if isinstance(node, ast.Call) and called_name(node=node.func) in _MEDIA_PART_CALLS:
        return [(keyword.arg, keyword.value) for keyword in node.keywords if keyword.arg]
    if isinstance(node, ast.Dict):
        fields = [
            (key.value, value)
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        part_type = dict(fields).get("type")
        if isinstance(part_type, ast.Constant) and part_type.value in _MEDIA_PART_TYPES:
            return fields
    return None


def _offending_arguments(fields: list[tuple[str, ast.expr]]) -> list[str]:
    """Returns the media-source fields of a part that are not a local data URI."""
    offenders: list[str] = []
    for name, value in fields:
        if name == "file_url":
            offenders.append("file_url")
        elif name == "image_url" and not (
            isinstance(value, ast.Call) and called_name(node=value.func) == _DATA_URI_BUILDER
        ):
            offenders.append("image_url")
    return offenders


def test_no_media_part_is_built_from_a_remote_url() -> None:
    """No media part in src/ carries a remote URL; media reaches the model via the Files API."""
    findings: list[str] = []
    inspected = 0
    for path in python_modules(root=PACKAGE):
        tree = ast.parse(source=path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.expr):
                continue
            fields = _media_part_fields(node=node)
            if fields is None:
                continue
            inspected += 1
            findings.extend(
                f"{path.relative_to(PACKAGE.parent)}:{node.lineno} passes {argument}"
                for argument in _offending_arguments(fields=fields)
            )
    assert inspected, "the scan read no media part, so it could not have flagged one"
    assert findings == [], (
        "media parts must reference an uploaded Files API uri via file_id, not a remote URL "
        f"(see the module docstring): {findings}"
    )
