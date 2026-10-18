"""The model tiers every runtime call dispatches on, and the shapes a triage call fills in."""

from typing import Literal, cast
from datetime import UTC, datetime

from pydantic import Field, BaseModel, computed_field
from openai.types.responses.tool_param import ToolParam
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning

from discordbot.typings.emojis import LinkSourceName


class ModelSettings(BaseModel):
    """Model name and reasoning effort that should be used together."""

    name: str = Field(
        ...,
        description="LiteLLM model string dispatched on the Responses API.",
        examples=["gemini-3.8-flash", "gemini-3.1-flash-image"],
    )
    # Which efforts a model accepts is per-model and not derivable from its name, its family, or
    # a sibling snapshot. The vocabulary itself differs by provider, so a guess is not even wrong
    # in a consistent direction: Gemini has `minimal` and no `xhigh`, OpenAI has `xhigh` and on
    # some models `none`, Anthropic has `max`. Look the model up in openrouter's list
    # (https://openrouter.ai/api/v1/models) under its `<provider>/<name>` id, where `reasoning`
    # carries `supported_efforts`, `default_effort` and `mandatory` (whether thinking can be
    # switched off at all). Sending an effort outside a model's own set is never a downgrade: direct
    # to Google it is a hard failure, and through the proxy a non-streaming call is answered by the
    # fallback deployment instead, so do that lookup instead of trusting a list in this tree.
    #
    # Every tier this catalog ships today is Gemini, where `mandatory` is true and `none` /
    # `disable` are therefore never legal. LiteLLM does not reject them but rewrites them to
    # minimal / low with `includeThoughts: False`, silently ending the reasoning summary the
    # streaming preview is built on; on a `*-latest` alias, which carries no `gemini-3`
    # substring, it instead sends `thinkingBudget: 0`, which a Gemini 3.x model refuses. Both
    # are facts about these models and this proxy, not about the field.
    #
    # The `minimal` default serves only the tiers that dispatch no effort at all (the image /
    # video / music / TTS / agent models below); every conversational tier states its own.
    effort: ReasoningEffort = Field(
        default="minimal",
        description="Reasoning effort passed to the Responses API for this model.",
    )

    @property
    def reasoning(self) -> Reasoning:
        """Returns Responses API reasoning options with this model's effort and auto summary."""
        return Reasoning(effort=self.effort, summary="auto")

    @property
    def is_gemini(self) -> bool:
        """Returns whether `name` identifies a Gemini model (`"gemini"` in `name`)."""
        return "gemini" in self.name

    @property
    def tools(self) -> list[ToolParam]:
        """Returns the built-in search and URL-grounding tool payloads for this model's provider.

        Code execution is intentionally omitted: Gemini and Claude validate every
        uploaded file part against code execution's narrow MIME allowlist and 400 the
        whole request on video, audio, or GIF-as-video attachments, so it cannot coexist
        with attachment ingestion.
        """
        if self.is_gemini:
            return cast("list[ToolParam]", [{"googleSearch": {}}, {"urlContext": {}}])
        if "claude" in self.name:
            return cast(
                "list[ToolParam]",
                [
                    {"type": "web_search_20260209", "name": "web_search"},
                    {"type": "web_fetch_20260209", "name": "web_fetch"},
                ],
            )
        if "grok" in self.name:
            return cast("list[ToolParam]", [{"type": "web_search"}, {"type": "x_search"}])
        return [{"type": "web_search"}]


class RuntimeModelCatalog(BaseModel):
    """Runtime model settings used by Discord bot LLM paths."""

    @computed_field
    @property
    def is_peak(self) -> bool:
        """Returns whether the current UTC time is inside the weekday peak-hour window (08:00-17:00).

        No tier reads this today. Kept on purpose: peak-hour high-demand refusals come and go,
        and this is what a tier branches on when they return.
        """
        now = datetime.now(tz=UTC)
        return now.weekday() < 5 and 8 <= now.hour < 17

    @property
    def image_model(self) -> ModelSettings:
        """Returns model settings for `images.generate` and `images.edit`."""
        return ModelSettings(name="gemini-nano-banana-2.1")

    @property
    def video_model(self) -> ModelSettings:
        """Returns model settings for video generation and source-video editing (`task="edit"`).

        Dispatched on the native Gemini Interactions API (`interactions.create`) direct to
        Google with a bare model name and no provider prefix.
        """
        return ModelSettings(name="gemini-omni-1.1-flash")

    @property
    def music_model(self) -> ModelSettings:
        """Returns model settings for music generation on the native Lyria Interactions API."""
        return ModelSettings(name="lyria-3.5")

    @property
    def tts_model(self) -> ModelSettings:
        """Returns model settings for spoken-reply synthesis on the `audio.speech` endpoint.

        Only the reply's `<generate-voice>` segments are synthesized into one clip;
        `effort` is unused for TTS.
        """
        return ModelSettings(name="gemini-3.8-flash-tts")

    @property
    def antigravity_model(self) -> ModelSettings:
        """Returns model settings for the deep-research Antigravity managed agent.

        Dispatched directly on the Gemini Interactions API; `effort` and `tools` are unused
        because the agent runs its own internal tool loop.
        """
        return ModelSettings(name="antigravity-preview-09-2026")

    @property
    def antigravity_reasoning_model(self) -> ModelSettings:
        """Returns the reasoning model settings passed as `agent_config.model` for deep research.

        Must name a model accepted by the Antigravity agent; `effort` and `tools` are unused.
        """
        return ModelSettings(name="gemini-3.8-flash")

    @property
    def triage_model(self) -> ModelSettings:
        """Returns model settings for filling a fixed-schema triage output.

        Confirm any repointed snapshot supports `minimal` before carrying it across (see
        `ModelSettings.effort`).
        """
        return ModelSettings(name="gemini-3.5-flash-lite", effort="minimal")

    @property
    def fast_model(self) -> ModelSettings:
        """Returns model settings for intermediate prose generation short of the final answer."""
        return ModelSettings(name="gemini-3.8-flash", effort="medium")

    @property
    def slow_model(self) -> ModelSettings:
        """Returns model settings for full text replies and strategic reasoning.

        Callers read `name` directly for provider and modality checks as well as dispatching
        it, so repointing this tier changes what reaches the model, not just how well it
        reasons.
        """
        # Pinned to a snapshot, never a `*-latest` alias: this tier's effort is the route's grade,
        # which the YouTube answer turn hands to `interactions.create` as a `thinking_level`, and
        # a level the model does not list loses the whole reply there and errors the
        # streaming answer through the proxy. Before repointing, check the new snapshot's set
        # (see `ModelSettings.effort`) covers every value `RouteClassification.effort` emits.
        # `gemini-4-argon` accepts low / medium / high, not `minimal` (Google's thinking page,
        # read 2026-10-14; openrouter does not list it).
        #
        # Without an entry in LiteLLM's price table a reply's footer shows no cost, and a
        # non-Gemini name falls back to the `{"text", "image"}` modality gate: an audio or video
        # attachment then never reaches the route or the answer model, so a clip posted with one
        # line of text is answered as if the line were the whole message.
        return ModelSettings(name="gemini-4-argon", effort="high")

    @property
    def memory_writer_model(self) -> ModelSettings:
        """Returns model settings for background long-term memory evaluation and consolidation.

        The note evaluator and the consolidator share one tier because both gate what is
        staged and written to fact files and the tone note; a weaker model on either risks
        losing or leaking memory.
        """
        return ModelSettings(name="gemini-4-argon", effort="high")


class RouteClassification(BaseModel):
    """Structured reply-mode classification returned by the route model."""

    decision: Literal["IMAGE", "VIDEO", "QA"] = Field(
        ..., description="Reply mode selected for the incoming Discord message."
    )
    watch_video: bool = Field(
        default=False,
        description=(
            "Set true only when the message links a YouTube video AND the user wants its "
            "content analyzed, summarized, or asked about; false when the link is incidental. "
            "Consumed only on the QA route to decide whether to watch the video."
        ),
    )
    link_context_sources: list[LinkSourceName] = Field(
        default_factory=list,
        description=(
            "Registered linked-post sources whose actual content the user wants analyzed, "
            "summarized, or discussed. Empty when every matching link is incidental. Consumed "
            "only on the QA route to decide which source builders may start."
        ),
    )
    # Overrides the answer model's effort on the QA path. Deliberately binary, with `high` as the
    # grade an ordinary message gets and `low` as the exception that has to be earned: the route
    # reads text-only parts, so it never sees an attachment's content, a linked post, or the
    # history behind a short message, and every one of those blind spots hides work rather than
    # inventing it. Both values still have to exist on whatever `slow_model` names, and no effort
    # is universal — not even `high` — so repointing that tier or widening this grade is checked
    # against the snapshot first (see `ModelSettings.effort`).
    effort: Literal["low", "high"] = Field(
        default="high",
        description=(
            "Reasoning effort the answer model should spend. Use low only for a message "
            "that asks for nothing, looks nothing up and works nothing out — banter and "
            "greetings, even when phrased as a question — and that is fully answerable "
            "from what you were shown; everything else, including anything you are "
            "unsure about, is high."
        ),
    )


# The route's schema on a turn that offers optional recall candidates. No docstring on purpose:
# under `text_format=` a class docstring is sent as the schema's description, and this shape was
# measured without one.
class RecallRouteClassification(RouteClassification):
    recall_user_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ids from the additional-members block whose long-term memory the latest message "
            "obliquely refers to. Empty is the normal case."
        ),
    )
