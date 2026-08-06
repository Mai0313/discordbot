"""Runtime configuration for the Douyin link features.

Holds `DouyinConfig` alone: the environment-backed switch `parse_douyin/cog.py` builds once in
its `__init__` and reads last in `on_message`, after the bot-author, URL-match, post-shape and
addressed-to-the-bot gates have each had their chance to return. So it is read only on a human,
non-addressed message carrying a Douyin post URL; a bot's own post, or anyone handing the bot a
post link, never reaches it. Nothing else keys off this module. The reply pipeline's own Douyin
switch (`douyin_video_enabled`, which gates whether the answer model is handed a linked clip to
watch) lives with the model kill-switches in `typings/llm.py`, and the two cover different paths
rather than overlapping: a link addressed to the bot is answered about and never expanded, so
neither switch turns the other one off.

`dotenv.load_dotenv()` runs at import here, as in the other config modules, so a `DouyinConfig()`
built anywhere reads the deployment's `.env`. `tests/test_parse_douyin_cog.py` therefore
substitutes an explicit stub for the cog's config instead of trusting the environment: a dev box
with the switch turned off would otherwise quietly reduce the tests that assert nothing happened
to no-ops that still pass. The positive ones fail loudly instead, on a missing reply or an empty
reaction list, so the stub is what keeps the vacuous half honest rather than the whole file.
"""

import dotenv
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv.load_dotenv()


class DouyinConfig(BaseSettings):
    """Configuration for Douyin link auto-expansion, read from environment variables.

    Kept apart from `LLMConfig` because auto-expansion is not an LLM feature: it downloads a
    post and posts it back. The reply pipeline's own Douyin switch lives with the other model
    kill-switches instead.

    Attributes:
        auto_expand_enabled: Kill-switch for expanding a pasted Douyin link into the channel.
            The one lever that stops the bot talking to Douyin at all if its WAF starts
            blocking, since expansion turns every pasted link into a request.
    """

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    auto_expand_enabled: bool = Field(
        default=True,
        description="Whether a Douyin link pasted in chat is expanded into the channel.",
        validation_alias=AliasChoices("DOUYIN_AUTO_EXPAND_ENABLED"),
    )
