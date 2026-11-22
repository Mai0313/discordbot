"""The linked-content sources `gen_reply` reads into answer context, in splice order.

The blocks land in the answer input in `LINK_CONTEXT_SOURCES` order, just before the current
message. Adding a source is one entry here, its builder module beside this one, its name in
`typings/emojis.py::LinkSourceName` (a source the router cannot name is never selected, so its
builder never starts) with a marker in `LINK_SOURCE_EMOJIS`, and a line in `route_prompt` saying
when the router should select it; the pipeline loops stay untouched.

Each entry is a thin adapter over its builder function rather than the function itself: an
adapter body resolves the builder name from this module's globals at call time, so a test
monkeypatching `discordbot.cogs.gen_reply.link_sources.registry.build_*_context_messages` still
intercepts the call.
"""

from google import genai
from openai.types.responses.response_input_param import EasyInputMessageParam

from discordbot.typings.llm import LLMConfig
from discordbot.services.platforms.douyin import DOUYIN_POST_LINK
from discordbot.services.platforms.threads import THREADS_POST_LINK
from discordbot.services.platforms.twitter import TWITTER_POST_LINK
from discordbot.cogs.gen_reply.link_sources import LinkContextSource
from discordbot.services.platforms.bilibili import BILIBILI_POST_LINK
from discordbot.services.platforms.facebook import FACEBOOK_POST_LINK
from discordbot.services.platforms.instagram import INSTAGRAM_POST_LINK
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
    url: str, media_client: genai.Client | None, deadline: float
) -> list[EasyInputMessageParam]:
    """Adapts the Threads builder to the registry signature (a straight pass-through)."""
    return await build_threads_context_messages(
        url=url, media_client=media_client, deadline=deadline
    )


async def _build_douyin_link_context(
    url: str, media_client: genai.Client | None, deadline: float
) -> list[EasyInputMessageParam]:
    """Adapts the Douyin builder to the registry signature (a straight pass-through)."""
    return await build_douyin_context_messages(
        url=url, media_client=media_client, deadline=deadline
    )


async def _build_bilibili_link_context(
    url: str, media_client: genai.Client | None, deadline: float
) -> list[EasyInputMessageParam]:
    """Adapts the Bilibili builder to the registry signature (a straight pass-through)."""
    return await build_bilibili_context_messages(
        url=url, media_client=media_client, deadline=deadline
    )


async def _build_facebook_link_context(
    url: str, media_client: genai.Client | None, deadline: float
) -> list[EasyInputMessageParam]:
    """Adapts the Facebook builder to the registry signature (a straight pass-through)."""
    return await build_facebook_context_messages(
        url=url, media_client=media_client, deadline=deadline
    )


async def _build_twitter_link_context(
    url: str, media_client: genai.Client | None, deadline: float
) -> list[EasyInputMessageParam]:
    """Adapts the Twitter builder to the registry signature (a straight pass-through)."""
    return await build_twitter_context_messages(
        url=url, media_client=media_client, deadline=deadline
    )


async def _build_instagram_link_context(
    url: str, media_client: genai.Client | None, deadline: float
) -> list[EasyInputMessageParam]:
    """Adapts the Instagram builder to the registry signature (a straight pass-through)."""
    return await build_instagram_context_messages(
        url=url, media_client=media_client, deadline=deadline
    )


def _needs_files_api(config: LLMConfig) -> bool:
    """Returns whether the Gemini Files API is enabled and configured with an API key."""
    return config.file_api_enabled and config.gemini_key_configured


def _douyin_media_ingest_allowed(config: LLMConfig) -> bool:
    """Returns whether Douyin media download and Files API upload are enabled."""
    return config.douyin_video_enabled and _needs_files_api(config=config)


def _bilibili_media_ingest_allowed(config: LLMConfig) -> bool:
    """Returns whether Bilibili media download and Files API upload are enabled."""
    return config.bilibili_video_enabled and _needs_files_api(config=config)


# Sources with `search_replied_to_message=True` fetch comments/replies omitted by auto-expansion.
LINK_CONTEXT_SOURCES: tuple[LinkContextSource, ...] = (
    LinkContextSource(
        name="threads",
        post_link=THREADS_POST_LINK,
        search_replied_to_message=True,
        build=_build_threads_link_context,
        timeout_notice=THREADS_TIMEOUT_NOTICE,
        media_ingest_allowed=_needs_files_api,
    ),
    LinkContextSource(
        name="facebook",
        post_link=FACEBOOK_POST_LINK,
        search_replied_to_message=True,
        build=_build_facebook_link_context,
        timeout_notice=FACEBOOK_TIMEOUT_NOTICE,
        media_ingest_allowed=_needs_files_api,
    ),
    LinkContextSource(
        name="instagram",
        post_link=INSTAGRAM_POST_LINK,
        search_replied_to_message=True,
        build=_build_instagram_link_context,
        timeout_notice=INSTAGRAM_TIMEOUT_NOTICE,
        media_ingest_allowed=_needs_files_api,
    ),
    LinkContextSource(
        name="twitter",
        post_link=TWITTER_POST_LINK,
        search_replied_to_message=True,
        build=_build_twitter_link_context,
        timeout_notice=TWITTER_TIMEOUT_NOTICE,
        media_ingest_allowed=_needs_files_api,
    ),
    LinkContextSource(
        name="douyin",
        post_link=DOUYIN_POST_LINK,
        build=_build_douyin_link_context,
        timeout_notice=DOUYIN_TIMEOUT_NOTICE,
        media_ingest_allowed=_douyin_media_ingest_allowed,
    ),
    LinkContextSource(
        name="bilibili",
        post_link=BILIBILI_POST_LINK,
        build=_build_bilibili_link_context,
        timeout_notice=BILIBILI_TIMEOUT_NOTICE,
        media_ingest_allowed=_bilibili_media_ingest_allowed,
    ),
)
