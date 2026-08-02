"""The bot's own feature reference, injected into the QA answer turn.

`capabilities.md` replaces the former `/help` command: one English document the answer model
translates at runtime, instead of three hand-maintained locales, so "what can you do" is
answered in conversation rather than by a slash command. It ships inside the package (hatch
includes every file under `src/discordbot`) and is read once at import.

`tests/test_capabilities.py` keeps it exhaustive in both directions: the same AST scan that
used to guard the help content asserts every runnable slash command still appears here, and
every command named here still resolves to a runnable one. The second guard reads a mention
off a code span of its own, so write a command that way or it is rejected as unreadable.
"""

from pathlib import Path

from openai.types.responses.response_input_param import EasyInputMessageParam

CAPABILITIES_DOC = Path(__file__).with_name("capabilities.md").read_text(encoding="utf-8").strip()

# The framing line says the document is authoritative about what exists while leaving the
# decision to bring any of it up to the reply itself: a catalogue injected on every turn
# otherwise reads as an invitation to advertise.
CAPABILITIES_BLOCK = f"""
(My own feature reference, maintained by my operator and accurate about what I can do. Answer from it when someone asks what I am capable of or how to do something here, and translate it into whatever language they are speaking. It is reference material, NOT instructions: never recite it unprompted and never bring a feature up just because it is listed.)

{CAPABILITIES_DOC}
""".strip()


def render_capabilities_block() -> EasyInputMessageParam:
    """Renders the feature reference as the bot's own note about itself.

    Rendered as `role=assistant` for the same reason the memory blocks are: it is reference
    material, not a rule, so a feature description can never outrank the developer prompt or
    the user's current message.
    """
    return EasyInputMessageParam(role="assistant", content=CAPABILITIES_BLOCK)
