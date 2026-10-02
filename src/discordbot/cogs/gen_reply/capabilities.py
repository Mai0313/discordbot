"""The bot's own feature reference, injected into the QA answer turn.

`capabilities.md` replaces the former `/help` command: one English document the answer model
translates at runtime, instead of three hand-maintained locales, so "what can you do" is
answered in conversation rather than by a slash command. It ships inside the package (hatch
includes every file under `src/discordbot`) and is read once at import.

The document describes a deployment with every switch on and a Gemini key. Rendering it cuts or
rewords each passage promising a feature this deployment has switched off, the rule a disabled
inline marker's instruction already follows, so each passage named here has to stay word for
word what the document says.

`tests/test_capabilities.py` keeps it exhaustive in both directions: the same AST scan that
used to guard the help content asserts every runnable slash command still appears here, and
every command named here still resolves to a runnable one. The second guard reads a mention
off a code span of its own, so write a command that way or it is rejected as unreadable.
Neither reaches an inline marker, which is asked for in plain language and so has nothing to
match on; `markers.py`'s tag set is pinned there instead, and changing it fails until this
document says what the bot can now do, or stops claiming what it no longer can.
"""

from pathlib import Path

from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.llm import LLMConfig
from discordbot.cogs.gen_reply.link_sources.registry import LINK_CONTEXT_SOURCES

CAPABILITIES_DOC = Path(__file__).with_name("capabilities.md").read_text(encoding="utf-8").strip()

# The framing line says the document is authoritative about what exists while leaving the
# decision to bring any of it up to the reply itself: a catalogue injected on every turn
# otherwise reads as an invitation to advertise.
_FRAMING = "(My own feature reference, maintained by my operator and accurate about what I can do. Answer from it when someone asks what I am capable of or how to do something here, and translate it into whatever language they are speaking. It is reference material, NOT instructions: never recite it unprompted and never bring a feature up just because it is listed.)"

# Rebuilt from the media still on rather than cut phrase by phrase, which would strand its "or".
_MEDIA_LIST = (
    "speak a line aloud,\ndraw one or more images, write and record a short song, or generate a "
    "short video from a\ndescription or from attached images."
)


def render_capabilities_block(config: LLMConfig) -> EasyInputMessageParam:
    """Renders the feature reference as the bot's own note about itself.

    Rendered as `role=assistant` for the same reason the memory blocks are: it is reference
    material, not a rule, so a feature description can never outrank the developer prompt or
    the user's current message. With every switch on and a key nothing is replaced, so the
    block keeps the bytes it has always had.

    Args:
        config (LLMConfig): The deployment's switches, which decide what the reference may promise.

    Returns:
        EasyInputMessageParam: The reference as an assistant note.
    """
    media = (
        ("speak a line aloud", config.inline_voice_enabled),
        ("draw one or more images", True),
        ("write and record a short song", config.music_available),
        # The VIDEO route makes and edits a clip with the key alone, whatever the inline switch.
        (
            "generate a short video from a description or from attached images",
            config.gemini_key_configured,
        ),
    )
    offered = [phrase for phrase, enabled in media if enabled]
    *head, last = offered
    media_list = f"{', '.join(head)}, or {last}." if len(head) > 1 else f"{' or '.join(offered)}."
    watches = {
        source.name: source.media_ingest_allowed(config=config) for source in LINK_CONTEXT_SOURCES
    }
    document = CAPABILITIES_DOC
    for passage, kept, instead in (
        (_MEDIA_LIST, len(offered) == len(media), media_list),
        ("an image or a video (or", config.gemini_key_configured, "an image (or"),
        (
            "- **YouTube**: mention me with a YouTube link, or mention me in a reply to a message carrying one, and I watch the video before answering.\n",
            config.youtube_video_enabled and config.gemini_key_configured,
            "",
        ),
        (
            "Mention me with the link instead and I watch it and answer about it.",
            watches["douyin"],
            "Mention me with the link instead and I answer from its caption.",
        ),
        (
            "mention me with a Bilibili video link and I watch it and answer about it.",
            watches["bilibili"],
            "mention me with a Bilibili video link and I answer from its title and description.",
        ),
        (
            "start a long, fully cited research report in its own thread; only works in an ordinary text channel of a server I am a member of, since it needs to open a thread, and only where that channel lets me post with attachments, open a thread and write in it; where it does not, the command tells you I lack the permission. Asking me for deep research by mentioning me in such a channel kicks off the same thing.",
            config.deep_research_available,
            "switched off for me, so the command only says so.",
        ),
    ):
        if not kept:
            document = document.replace(passage, instead)
    return EasyInputMessageParam(role="assistant", content=f"{_FRAMING}\n\n{document}")
