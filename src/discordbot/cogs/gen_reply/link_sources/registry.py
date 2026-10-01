"""The linked-content sources `gen_reply` reads into answer context, in splice order.

The blocks land in the answer input in `LINK_CONTEXT_SOURCES` order, just before the current
message. Adding a source is one entry here, its builder module beside this one, its name in
`typings/emojis.py::LinkSourceName` (a source the router cannot name is never selected, so its
builder never starts) with a marker in `LINK_SOURCE_EMOJIS`, and a line in `route_prompt` saying
when the router should select it; the pipeline loops stay untouched.

Each entry is a thin adapter over its builder function rather than the function itself: an
adapter body resolves the builder name from THIS module's globals at call time, so a test
monkeypatching `discordbot.cogs.gen_reply.link_sources.registry.build_*_context_messages` still
intercepts the call.
"""

from google import genai
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.llm import LLMConfig
from discordbot.services.platforms.douyin import DOUYIN_URL_RE, is_douyin_post_url
from discordbot.services.platforms.threads import THREADS_URL_RE
from discordbot.services.platforms.twitter import TWITTER_URL_RE
from discordbot.cogs.gen_reply.link_sources import LinkContextSource
from discordbot.services.platforms.bilibili import BILIBILI_URL_RE
from discordbot.services.platforms.facebook import FACEBOOK_URL_RE, is_facebook_post_url
from discordbot.services.platforms.instagram import INSTAGRAM_URL_RE, is_instagram_post_url
from discordbot.cogs.gen_reply.link_sources.douyin import (
    DOUYIN_TIMEOUT_NOTICE,
    build_douyin_context_messages,
)
from discordbot.cogs.gen_reply.link_sources.threads import (
    THREADS_TIMEOUT_NOTICE,
    build_threads_context_messages,
)
from discordbot.cogs.gen_reply.link_sources.twitter import (
    TWITTER_TIMEOUT_NOTICE,
    build_twitter_context_messages,
)
from discordbot.cogs.gen_reply.link_sources.bilibili import (
    BILIBILI_TIMEOUT_NOTICE,
    build_bilibili_context_messages,
)
from discordbot.cogs.gen_reply.link_sources.facebook import (
    FACEBOOK_TIMEOUT_NOTICE,
    build_facebook_context_messages,
)
from discordbot.cogs.gen_reply.link_sources.instagram import (
    INSTAGRAM_TIMEOUT_NOTICE,
    build_instagram_context_messages,
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


async def _build_facebook_link_context(
    *,
    url: str,
    answer_model_is_gemini: bool,
    gemini_client: genai.Client | None,
    allow_media_ingest: bool,
) -> list[EasyInputMessageParam]:
    """Adapts the Facebook builder to the registry signature (a straight pass-through)."""
    return await build_facebook_context_messages(
        url=url,
        answer_model_is_gemini=answer_model_is_gemini,
        gemini_client=gemini_client,
        allow_media_ingest=allow_media_ingest,
    )


async def _build_twitter_link_context(
    *,
    url: str,
    answer_model_is_gemini: bool,
    gemini_client: genai.Client | None,
    allow_media_ingest: bool,
) -> list[EasyInputMessageParam]:
    """Adapts the Twitter builder to the registry signature (a straight pass-through)."""
    return await build_twitter_context_messages(
        url=url,
        answer_model_is_gemini=answer_model_is_gemini,
        gemini_client=gemini_client,
        allow_media_ingest=allow_media_ingest,
    )


async def _build_instagram_link_context(
    *,
    url: str,
    answer_model_is_gemini: bool,
    gemini_client: genai.Client | None,
    allow_media_ingest: bool,
) -> list[EasyInputMessageParam]:
    """Adapts the Instagram builder to the registry signature (a straight pass-through)."""
    return await build_instagram_context_messages(
        url=url,
        answer_model_is_gemini=answer_model_is_gemini,
        gemini_client=gemini_client,
        allow_media_ingest=allow_media_ingest,
    )


def _threads_media_ingest_allowed(config: LLMConfig) -> bool:
    """Threads media ingestion has no kill-switch; the Gemini checks alone gate it."""
    del config
    return True


def _needs_files_api(config: LLMConfig) -> bool:
    """A source with no kill-switch of its own: the Files API and its key are the whole gate.

    `file_api_enabled` belongs here rather than only at the upload, because the media is fetched
    and downscaled before the upload it could no longer feed, so gating at the upload alone would
    still spend that work on the reply's critical path.
    """
    return config.file_api_enabled and config.gemini_key_configured


def _douyin_media_ingest_allowed(config: LLMConfig) -> bool:
    """The Douyin kill-switch on top of the shared Files API gate.

    The clip is downloaded first, on a WAF-sensitive path, so the switch has to be read before
    the fetch rather than at the upload.
    """
    return config.douyin_video_enabled and _needs_files_api(config=config)


def _bilibili_media_ingest_allowed(config: LLMConfig) -> bool:
    """The Bilibili kill-switch on top of the shared Files API gate.

    Same reason as Douyin minus the WAF: a 30-minute video is downloaded in full before the
    upload it can no longer feed.
    """
    return config.bilibili_video_enabled and _needs_files_api(config=config)


LINK_CONTEXT_SOURCES: tuple[LinkContextSource, ...] = (
    LinkContextSource(
        name="threads",
        url_pattern=THREADS_URL_RE,
        # Reads a link the user only replied to: what it fetches is the discussion under the
        # post, which the `parse_threads` expansion deliberately does not show, so
        # "@bot 這篇底下在吵什麼" on someone else's link has nothing else to answer from.
        search_replied_to_message=True,
        build=_build_threads_link_context,
        timeout_notice=THREADS_TIMEOUT_NOTICE,
        media_ingest_allowed=_threads_media_ingest_allowed,
    ),
    LinkContextSource(
        name="facebook",
        url_pattern=FACEBOOK_URL_RE,
        # The regex matches the host, not the path, so a profile or group home page would
        # otherwise spend a full ~950KB page fetch to establish there is no post.
        url_filter=is_facebook_post_url,
        # Reads a link the user only replied to: what it reads includes the comments under the
        # post, which the `parse_facebook` expansion deliberately does not show, so asking the
        # bot about someone else's linked post has something to answer from.
        search_replied_to_message=True,
        build=_build_facebook_link_context,
        timeout_notice=FACEBOOK_TIMEOUT_NOTICE,
        media_ingest_allowed=_needs_files_api,
    ),
    LinkContextSource(
        name="instagram",
        url_pattern=INSTAGRAM_URL_RE,
        # The regex matches the host, not the path, so a profile or the home page would
        # otherwise spend a full page fetch to establish there is no post.
        url_filter=is_instagram_post_url,
        # Reads a link the user only replied to: what it reads is the comment section, which the
        # `parse_instagram` expansion deliberately does not show.
        search_replied_to_message=True,
        build=_build_instagram_link_context,
        timeout_notice=INSTAGRAM_TIMEOUT_NOTICE,
        media_ingest_allowed=_needs_files_api,
    ),
    LinkContextSource(
        name="twitter",
        # Path-anchored on `/status/<digits>`, so no `url_filter` is needed: a profile or the
        # home page never matches in the first place.
        url_pattern=TWITTER_URL_RE,
        # Deliberately leaves `search_replied_to_message` off. A post source sets it when what it
        # fetches includes the comments its own expansion does not show, so a mention on someone
        # else's link has something new to answer from. Twitter's endpoint serves no replies at
        # all, so a second read of the same link would find exactly what the first did.
        build=_build_twitter_link_context,
        timeout_notice=TWITTER_TIMEOUT_NOTICE,
        media_ingest_allowed=_needs_files_api,
    ),
    LinkContextSource(
        name="douyin",
        url_pattern=DOUYIN_URL_RE,
        # The regex matches the host, not the path: a profile or live-room link is not a post,
        # so reading it would only spend a rate-limited Douyin request to say so.
        url_filter=is_douyin_post_url,
        build=_build_douyin_link_context,
        timeout_notice=DOUYIN_TIMEOUT_NOTICE,
        media_ingest_allowed=_douyin_media_ingest_allowed,
    ),
    LinkContextSource(
        name="bilibili",
        # Path-anchored to the watchable /video/ forms (plus b23.tv short links), so no url_filter
        # is needed on top.
        url_pattern=BILIBILI_URL_RE,
        build=_build_bilibili_link_context,
        timeout_notice=BILIBILI_TIMEOUT_NOTICE,
        media_ingest_allowed=_bilibili_media_ingest_allowed,
    ),
)
