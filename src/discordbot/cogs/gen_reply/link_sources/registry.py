"""The linked-content sources `gen_reply` reads into answer context, in splice order.

The blocks land in the answer input in `LINK_CONTEXT_SOURCES` order, just before the current
message. Adding a source is one entry here, its builder module beside this one, and its name in
`RouteClassification.link_context_sources` (a source the router cannot name is never selected, so
its builder never starts); the pipeline loops stay untouched.

Each entry is a thin adapter over its builder function rather than the function itself: an
adapter body resolves the builder name from THIS module's globals at call time, so a test
monkeypatching `discordbot.cogs.gen_reply.link_sources.registry.build_*_context_messages` still
intercepts the call.
"""

from google import genai
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.llm import LLMConfig
from discordbot.utils.douyin import DOUYIN_URL_RE, is_douyin_post_url
from discordbot.utils.threads import THREADS_URL_RE
from discordbot.utils.bilibili import BILIBILI_URL_RE
from discordbot.cogs.gen_reply.link_sources import LinkContextSource
from discordbot.cogs.gen_reply.link_sources.douyin import (
    build_douyin_context_messages,
    douyin_timeout_context_messages,
)
from discordbot.cogs.gen_reply.link_sources.threads import (
    build_threads_context_messages,
    threads_timeout_context_messages,
)
from discordbot.cogs.gen_reply.link_sources.bilibili import (
    build_bilibili_context_messages,
    bilibili_timeout_context_messages,
)


async def _build_threads_link_context(
    *,
    url: str,
    answer_model_is_gemini: bool,
    gemini_client: genai.Client | None,
    allow_media_ingest: bool,
) -> list[EasyInputMessageParam]:
    """Adapts the Threads builder to the registry signature.

    Threads media ingestion has no kill-switch, so the flag is accepted and dropped.
    """
    del allow_media_ingest
    return await build_threads_context_messages(
        url=url, answer_model_is_gemini=answer_model_is_gemini, gemini_client=gemini_client
    )


async def _build_douyin_link_context(
    *,
    url: str,
    answer_model_is_gemini: bool,
    gemini_client: genai.Client | None,
    allow_media_ingest: bool,
) -> list[EasyInputMessageParam]:
    """Adapts the Douyin builder to the registry signature (a straight pass-through)."""
    return await build_douyin_context_messages(
        url=url,
        answer_model_is_gemini=answer_model_is_gemini,
        gemini_client=gemini_client,
        allow_media_ingest=allow_media_ingest,
    )


async def _build_bilibili_link_context(
    *,
    url: str,
    answer_model_is_gemini: bool,
    gemini_client: genai.Client | None,
    allow_media_ingest: bool,
) -> list[EasyInputMessageParam]:
    """Adapts the Bilibili builder to the registry signature (a straight pass-through)."""
    return await build_bilibili_context_messages(
        url=url,
        answer_model_is_gemini=answer_model_is_gemini,
        gemini_client=gemini_client,
        allow_media_ingest=allow_media_ingest,
    )


def _threads_media_ingest_allowed(config: LLMConfig) -> bool:
    """Threads media ingestion has no kill-switch; the Gemini checks alone gate it."""
    del config
    return True


def _douyin_media_ingest_allowed(config: LLMConfig) -> bool:
    """The Douyin kill-switch plus the direct-Gemini key its Files API upload needs.

    `file_api_enabled` belongs here rather than only at the upload: the clip is downloaded
    first and the upload skipped after, so gating it there alone would still spend the whole
    fetch on a WAF-sensitive path for media that can no longer reach the model.
    """
    return (
        config.douyin_video_enabled
        and config.file_api_enabled
        and bool(config.gemini_api_key.strip())
    )


def _bilibili_media_ingest_allowed(config: LLMConfig) -> bool:
    """The Bilibili kill-switch plus the direct-Gemini key its Files API upload needs.

    Carries `file_api_enabled` for the reason the Douyin predicate does, minus the WAF: a
    30-minute video is downloaded in full before the upload it can no longer feed.
    """
    return (
        config.bilibili_video_enabled
        and config.file_api_enabled
        and bool(config.gemini_api_key.strip())
    )


LINK_CONTEXT_SOURCES: tuple[LinkContextSource, ...] = (
    LinkContextSource(
        name="threads",
        url_pattern=THREADS_URL_RE,
        # The one source that reads a link the user only replied to: what it fetches is the
        # discussion under the post, which the `parse_threads` expansion deliberately does not
        # show, so "@bot 這篇底下在吵什麼" on someone else's link has nothing else to answer from.
        search_replied_to_message=True,
        build=_build_threads_link_context,
        on_timeout=threads_timeout_context_messages,
        media_ingest_allowed=_threads_media_ingest_allowed,
    ),
    LinkContextSource(
        name="douyin",
        url_pattern=DOUYIN_URL_RE,
        # The regex matches the host, not the path: a profile or live-room link is not a post,
        # so reading it would only spend a rate-limited Douyin request to say so.
        url_filter=is_douyin_post_url,
        build=_build_douyin_link_context,
        on_timeout=douyin_timeout_context_messages,
        media_ingest_allowed=_douyin_media_ingest_allowed,
    ),
    LinkContextSource(
        name="bilibili",
        # Path-anchored to the watchable /video/ forms (plus b23.tv short links), so unlike
        # Douyin no url_filter is needed on top.
        url_pattern=BILIBILI_URL_RE,
        build=_build_bilibili_link_context,
        on_timeout=bilibili_timeout_context_messages,
        media_ingest_allowed=_bilibili_media_ingest_allowed,
    ),
)
