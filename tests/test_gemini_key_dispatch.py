"""Guards the rule that a proxy dispatch names a deployment and a direct one names a model.

One reply runs on one Gemini key, because a Files API file is readable only by the project
that uploaded it and a request naming files from two keys fails outright. Through the LiteLLM
proxy that pin is the `-key<n>` suffix `ModelSettings.deployment_name` adds; direct to Google
there is no such deployment and the key rides on the client instead, so the bare `name` is
the only thing Google will accept.

Getting either side wrong is silent. A proxy call left on `name` reaches the pooled
deployment, which the proxy answers from whichever key it likes, and the reply's uploaded
files are then on the wrong project. A direct call handed a `deployment_name` asks Google for
a model that does not exist. Neither shows up in a review diff, so it is a lint.

The scan is deliberately a denylist on `.name` rather than a requirement of
`.deployment_name`: `VoiceGenerator` holds an already-resolved deployment string in a plain
`str` field, and a whitelist would either reject that or have to name it as an exception.
"""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"

# Trailing method paths that dispatch through the LiteLLM proxy, where the key pin belongs.
_PROXY_DISPATCHES = frozenset({
    ("responses", "create"),
    ("responses", "parse"),
    ("responses", "stream"),
    ("images", "generate"),
    ("images", "edit"),
    ("speech", "create"),
})
# Trailing method paths that go direct to Google, where a `-key<n>` name would be rejected.
_DIRECT_DISPATCHES = frozenset({("interactions", "create")})


def _attribute_path(node: ast.expr) -> tuple[str, ...]:
    """Returns the dotted attribute path of a call target, outermost segment last."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _model_keyword(call: ast.Call) -> ast.expr | None:
    """Returns the `model=` argument of a call, or None when it has none."""
    for keyword in call.keywords:
        if keyword.arg == "model":
            return keyword.value
    return None


def _reads_attribute(node: ast.expr, attribute: str) -> bool:
    """Whether the expression is a plain `<something>.<attribute>` read."""
    return isinstance(node, ast.Attribute) and node.attr == attribute


def _offenders(wanted: frozenset[tuple[str, str]], forbidden_attribute: str) -> list[str]:
    """Finds dispatches in `wanted` whose `model=` reads `forbidden_attribute`."""
    found: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(source=path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            segments = _attribute_path(node.func)
            if len(segments) < 2 or tuple(segments[-2:]) not in wanted:
                continue
            model = _model_keyword(call=node)
            if model is not None and _reads_attribute(node=model, attribute=forbidden_attribute):
                found.append(f"{path.relative_to(_SRC)}:{node.lineno} .{'.'.join(segments[-2:])}")
    return found


def test_no_proxy_dispatch_names_the_bare_model() -> None:
    """Every proxied call carries the key pin, so a reply never crosses two Gemini projects."""
    offenders = _offenders(wanted=_PROXY_DISPATCHES, forbidden_attribute="name")
    assert offenders == [], (
        "A proxy dispatch on `.name` reaches the pooled deployment, so the proxy answers it "
        "from whichever key it likes while the reply's uploaded files sit on another. Use "
        f"`.deployment_name`. Offenders: {offenders}"
    )


def test_no_direct_dispatch_names_a_deployment() -> None:
    """Google is asked for a model, never for one of the proxy's key-pinned deployments."""
    offenders = _offenders(wanted=_DIRECT_DISPATCHES, forbidden_attribute="deployment_name")
    assert offenders == [], (
        "`interactions.create` goes direct to Google, which has no `-key<n>` deployment; the "
        f"key rides on the client. Use `.name`. Offenders: {offenders}"
    )


def test_the_scan_finds_the_dispatches_it_is_guarding() -> None:
    """Guards the scan itself: a rename upstream would otherwise make both tests vacuous."""
    proxy_seen = _offenders(wanted=_PROXY_DISPATCHES, forbidden_attribute="deployment_name")
    direct_seen = _offenders(wanted=_DIRECT_DISPATCHES, forbidden_attribute="name")
    assert proxy_seen, "The scan found no proxy dispatch at all, so its guard proves nothing."
    assert direct_seen, "The scan found no direct dispatch at all, so its guard proves nothing."
