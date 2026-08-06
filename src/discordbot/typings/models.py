"""Which model every runtime LLM path dispatches on, and the structured outputs of the router.

`ModelSettings` is one tier's dispatch pair — the model string plus the reasoning effort sent
with it — and derives from those two fields the payloads a call site needs: `reasoning` for the
Responses API and `tools` for the provider's built-in search / url grounding. That name is the
LiteLLM model string on the proxied tiers, but a bare Google model name on the three
direct-to-Google ones (`video_model`, `music_model`, `antigravity_model`), which dispatch on
`interactions.create` and read `name` alone.
`RuntimeModelCatalog` is the tier table itself, one property per runtime purpose. It reads no
configuration, so a tier moves by editing this file rather than the environment, and every cog
that needs one builds its own `RuntimeModelCatalog()` (`gen_reply`, `research`, `stock`, `memory`,
`auto_unmute`). The dev scripts under `scripts/` are the exception and mostly hardcode their own
`ModelSettings` copies, so a tier change here has to be mirrored into them by hand or they go on
exercising a model production no longer runs.

Two constraints cut across the whole table. `minimal` is the effort floor and `none` is never
used: Gemini 3 cannot switch thinking off, and LiteLLM only recognises a model as Gemini 3 by the
literal `gemini-3` substring, which the `*-latest` aliases dispatched here do not carry, so `none`
falls through to the pre-3 branch and sends a `thinkingBudget: 0` a 3.x model rejects
(`tests/test_runtime_models.py` pins it, because the breakage shows only against the live API).
And an alias is suspect wherever a capability gate might key on the model string — `*-latest` has
already cost both an effort level and native built-in-plus-function tool combination — so prefer a
concrete snapshot on any tier whose capabilities matter.

`RouteClassification` and `EffortGrade` are the parsed outputs of the two parallel classification
calls (`_route_classify` / `_grade_effort` hand them to `responses.parse` as `text_format`). Their
field descriptions are the schema the route model actually reads, so editing that wording is
prompt work, not documentation.
"""

from typing import Literal, cast
from datetime import UTC, datetime

from pydantic import Field, BaseModel, computed_field
from openai.types.responses.tool_param import ToolParam
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning


class ModelSettings(BaseModel):
    """One tier's dispatch pair: the model string and the reasoning effort sent with it.

    `reasoning` and `tools` are derived from these two fields, so a call site holding a
    `ModelSettings` never re-derives the provider payloads. Override the effort for a single call
    with `model_copy(update={"effort": ...})`, as the answer turn does with the route-decided
    grade, rather than adding a tier.

    Attributes:
        name: LiteLLM model string dispatched on the Responses API.
        effort: Reasoning effort passed to the Responses API for this model.
    """

    name: str = Field(
        ...,
        description="LiteLLM model string dispatched on the Responses API.",
        examples=["gemini-flash-latest", "gemini-3.1-flash-image"],
    )
    # `minimal`, never `none`: Gemini 3 cannot switch thinking off, so its own vocabulary starts
    # there (`thinking_level` accepts minimal / low / medium / high). On the pre-3 LiteLLM branch
    # the `*-latest` aliases fall through to, `minimal` maps to a small positive budget a 3.x
    # model accepts, while `none` maps to the `thinkingBudget: 0` it rejects — which is how
    # `none` stopped working the moment `gemini-flash-latest` began resolving to a 3.x snapshot.
    effort: ReasoningEffort = Field(
        default="minimal",
        description="Reasoning effort passed to the Responses API for this model.",
    )

    @property
    def reasoning(self) -> Reasoning:
        """Responses API reasoning options for this model.

        `summary="auto"` is load-bearing rather than cosmetic: the streamer paints the returned
        thought summary as `-#` subtext until the first content delta, so dropping it would take
        the whole reasoning preview with it.

        Returns:
            Reasoning options carrying this model's configured effort and an automatic summary.
        """
        return Reasoning(effort=self.effort, summary="auto")

    @property
    def tools(self) -> list[ToolParam]:
        """The provider's built-in search / url grounding tools, selected by the model string.

        Code execution is intentionally omitted: Gemini and Claude validate every uploaded file
        part against code execution's narrow MIME allowlist and 400 the whole request on
        video / audio / GIF-as-video attachments, so it cannot coexist with the attachment
        ingestion path. Search / url grounding have no such limit.

        These are server-side tools only, and mixing a custom function tool in with them is a
        Gemini-branch question: there, that turn takes `include_server_side_tool_invocations:
        true` in its `extra_body`, or LiteLLM's Gemini transform strips the search tools before
        dispatch and grounding drops to zero silently, with no 400. No call site sets the flag
        today because none mixes one in. The flag is not the mechanism on the Claude, Grok and
        OpenAI branches, where that failure mode is simply unmeasured.

        Returns:
            googleSearch + urlContext for a Gemini model, web_search + web_fetch for Claude,
            web_search + x_search for Grok, and the OpenAI web_search tool for anything else.
        """
        if "gemini" in self.name:
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
        return cast("list[ToolParam]", [{"type": "web_search"}])


class RuntimeModelCatalog(BaseModel):
    """The runtime tier table: one property per LLM purpose the bot dispatches on.

    Fieldless and configuration-free, so every cog builds its own instance instead of sharing
    one. Each property below names its callers and the reason it sits at the tier it does; keep
    those lists in sync when runtime model usage moves.
    """

    @computed_field
    @property
    def is_peak(self) -> bool:
        """Whether runtime model selection is in the peak-hour window.

        Evaluated per read rather than cached, so a long-running process crosses the boundary
        without a restart. A `computed_field` so a serialized catalog states which branch
        `slow_model` was taken from, which is the only tier it decides.

        Returns:
            True during UTC weekdays from 08:00 up to (but excluding) 17:00, otherwise False.
        """
        now = datetime.now(UTC)
        return now.weekday() < 5 and 8 <= now.hour < 17

    @property
    def image_model(self) -> ModelSettings:
        """The model settings for image generation and editing.

        Callers: `ImageGenerator` (its `render` for the IMAGE route `_handle_image_reply`, its
        best-effort `generate` for the QA-route inline `<generate-image>` marker).

        Returns:
            Model settings used with `images.generate` and `images.edit`.
        """
        return ModelSettings(name="gemini-3.1-flash-image")

    @property
    def video_model(self) -> ModelSettings:
        """The model settings for video generation.

        Callers: `VideoGenerator.render` (via the VIDEO route `_handle_video_reply`).

        Returns:
            Model settings used with the native Gemini Interactions API (`interactions.create`,
            a bare model name with no provider prefix, since the call goes direct to Google not
            via the proxy). omni unifies text/image/reference/edit video generation, so the same
            model backs plain generation and true source-video editing (`task="edit"`).
        """
        return ModelSettings(name="gemini-omni-flash-preview")

    @property
    def music_model(self) -> ModelSettings:
        """The model settings for music generation.

        Callers: `MusicGenerator.generate` (via the QA-route inline `<generate-music>` marker).

        Returns:
            Model settings used with the native Gemini (Lyria) Interactions API (a bare model
            name, no provider prefix, since the call goes direct to Google not via the proxy).
        """
        return ModelSettings(name="lyria-3-clip-preview")

    @property
    def antigravity_model(self) -> ModelSettings:
        """The deep-research agent: a one-shot Antigravity managed agent.

        Callers: `ResearchCogs` (the only research tier there is).

        Returns:
            The Antigravity managed-agent string dispatched on the Gemini Interactions API
            (direct, not the proxy). `effort` / `tools` are unused on the agent path: the
            agent runs its own internal tool loop.
        """
        return ModelSettings(name="antigravity-preview-05-2026")

    @property
    def prompt_model(self) -> ModelSettings:
        """The model settings for the image/video generation prompt director.

        Callers: `PromptGenerator.refine` (via `_handle_image_reply`, `_handle_video_reply`).

        Returns:
            Flash-with-high-effort settings for the director call that expands a thin user
            request into a rich, self-contained generation prompt before the image/video model
            draws it. Flash (not flash-lite) so it reliably CALLS the grounding tools
            (googleSearch / urlContext) to look up named subjects; effort is the latency lever
            since this call sits serially on the IMAGE/VIDEO critical path before generation.
            Deliberately decoupled from `slow_model` so the refinement tier can be tuned on its
            own (bump to a pro snapshot here if the refined prompts underperform).
        """
        return ModelSettings(name="gemini-flash-latest", effort="high")

    @property
    def media_reply_model(self) -> ModelSettings:
        """The model settings for the conversational reply that rides generated media.

        Callers: `_stream_media_persona_reply` (via `_handle_image_reply` and
        `_handle_video_reply`).

        One tier for both routes because they are the same task on the same shared streamer:
        only the system prompt and the focus part differ per route. Flash, the middle tier
        between the flash-lite caption it replaces and the gemini-pro answer model, and it
        ingests video as well as images: it reads conversation history and the selected user
        memory, then answers in persona while holding the image or clip it just made rather
        than coldly describing it.

        Returns:
            Flash with `effort="low"`, which keeps it snappy yet still emits a reasoning
            summary so the streaming reasoning preview shows. The media is already on screen,
            so this text streams in after with no added generation latency.
        """
        return ModelSettings(name="gemini-flash-latest", effort="low")

    @property
    def tts_model(self) -> ModelSettings:
        """The model settings for spoken-reply text-to-speech.

        Callers: `VoiceGenerator` (via `ReplyGeneratorCogs.voice_generator`).

        Returns:
            Model settings whose name is dispatched on the `audio.speech` endpoint to render the
            reply's `<generate-voice>` spans into one clip. Only `name` is read: that endpoint
            takes no reasoning options, so `effort` is unused here.
        """
        return ModelSettings(name="gemini-3.1-flash-tts-preview")

    @property
    def fast_model(self) -> ModelSettings:
        """The model settings for lightweight, single-shot tasks.

        Callers: `_route_classify`, `_grade_effort`, `AutoUnmuteCogs._generate_reply`,
        `StockNewsAI`, the research thread title.

        This is the difficulty tier, not a purpose: everything routed here is one short call
        whose job is either a narrow classification or a throwaway line of text. The route
        picks the reply mode and the effort grade decides how hard the answer model should
        think; both follow simple rules, so flash-lite is enough and the QA critical path
        stays short (the grade runs in parallel with the route under the same `route_done`
        gate, so it costs nothing when it beats the route and at most the post-route
        `EFFORT_GRACE_SECONDS` grace when it does not).

        Returns:
            Fast minimal-thinking settings.
        """
        return ModelSettings(name="gemini-flash-lite-latest", effort="minimal")

    @property
    def tool_model(self) -> ModelSettings:
        """The model settings for optional oblique-reference memory selection.

        Callers: `_select_user_memories`.

        Returns:
            Fast minimal-thinking settings for matching an obliquely referenced absent
            member to the public nickname table: flash (not flash-lite)
            because matching spoken community nicknames to user ids needs more
            language skill than the lite tier reliably delivers, while staying far
            below answer-model latency.
        """
        return ModelSettings(name="gemini-flash-latest", effort="minimal")

    @property
    def slow_model(self) -> ModelSettings:
        """The model settings for full text replies and strategic reasoning.

        Callers: `_handle_message_reply` (which overrides `effort` with the
        route-decided level), attachment modality gating, and dev scripts.

        Its NAME is read as a capability signal well beyond the answer turn: whether it contains
        `gemini` picks the attachment handler (Files API upload vs per-type inlining) and decides
        whether a linked post's media is ingested at all, and `get_supported_modalities` gates
        which attachments survive. Moving this tier off Gemini therefore changes what the bot can
        read, not just how it writes.

        Returns:
            Slow-path model settings for reply generation and summaries, split peak / off-peak.
        """
        # Off-peak dispatches the `gemini-pro-latest` alias in full knowledge of the trap: Google
        # silently downgrades that alias to the gemini-3-pro generation, whose Interactions
        # `thinking_level` enum allows only low / high and rejects `medium`, where the explicit
        # gemini-3.1-pro-preview snapshot this used to pin accepts it. 2026-08-05: deliberately
        # back on the alias to re-measure whether that downgrade still happens; pin the snapshot
        # again if it does.
        # The peak / off-peak split is kept on purpose because Gemini Pro has historically slowed
        # down during peak hours, which is why peak takes flash instead.
        if self.is_peak:
            return ModelSettings(name="gemini-flash-latest", effort="high")
        return ModelSettings(name="gemini-pro-latest", effort="high")

    @property
    def memory_extractor_model(self) -> ModelSettings:
        """The model settings for phase-1 memory extraction.

        Callers: `MemoryExtractorAI.extract` (its `extract_model` field).

        Returns:
            Model settings for the background memory extraction call. Kept apart from the
            writer tier so the recall-oriented first pass can be downgraded on its own if
            the gates behind it prove able to carry the precision.
        """
        return ModelSettings(name="gemini-pro-latest", effort="high")

    @property
    def memory_writer_model(self) -> ModelSettings:
        """The model settings for everything deciding what reaches long-term memory.

        Callers: the phase-1.5 evaluator (`MemoryExtractorAI.extract`, its optional
        `evaluate_model` field) and phase-2 consolidation (`MemoryExtractorAI.consolidate`,
        its `consolidate_model` field), which also backs `regenerate_main_memory`, plus
        `scripts/migrate_memories.py`, which reads this tier to drive that rebuild offline.

        Returns:
            Model settings for the background memory write calls. One tier for both because
            both are gates on the same side: the evaluator is the last LLM check before an
            observation is staged, tightening `sharing` and `durability` (downgrade-only),
            deduping by `normalized_key` and stripping personal-attack wording, and the
            consolidator turns that staging into the fact files plus the tone note. A weaker
            model on either loses memory or leaks it, rather than just failing to record it.
        """
        return ModelSettings(name="gemini-pro-latest", effort="high")


class RouteClassification(BaseModel):
    """Structured reply-mode classification returned by the route model.

    The only critical-path classification: dispatch waits on it. An unparsable or empty output
    defaults to `QA`, and a `SUMMARY` carrying a URL is steered back to QA with both content-read
    fields preserved, so the corrected route still ingests what the user asked about. Those two
    fields are read on the QA route alone; every other route ignores them.

    Attributes:
        decision: The reply mode selected for the incoming Discord message.
        watch_video: Whether the QA answer should ingest a linked YouTube video.
        link_context_sources: Linked-post sources whose content the QA answer should ingest.
    """

    decision: Literal["IMAGE", "VIDEO", "QA", "SUMMARY"] = Field(
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
    link_context_sources: list[Literal["threads", "douyin", "bilibili"]] = Field(
        default_factory=list,
        description=(
            "Registered linked-post sources whose actual content the user wants analyzed, "
            "summarized, or discussed. Empty when every matching link is incidental. Consumed "
            "only on the QA route to decide which source builders may start."
        ),
    )


class EffortGrade(BaseModel):
    """Structured answer-effort grade returned by the effort model.

    Graded by a call that runs in parallel with the route, so it costs nothing when it finishes
    first; when it does not, `_resolve_effort` waits for it on the QA / SUMMARY critical path
    ahead of the answer turn, adding up to the post-route `EFFORT_GRACE_SECONDS` grace. The
    answer model's effort is overridden with it on the QA and SUMMARY paths, while IMAGE and
    VIDEO cancel the grading task. A timeout or failure falls back to `high`, this field's own
    default, so a lost grade degrades to the most expensive setting, never the cheapest.

    Attributes:
        effort: Reasoning effort the answer model should spend on this message.
    """

    effort: Literal["low", "medium", "high"] = Field(
        default="high",
        description=(
            "Reasoning effort the answer model should spend: high for any substantive "
            "question or task, medium for trivial lookups or transforms, low only for "
            "pure social chatter."
        ),
    )


__all__ = ["EffortGrade", "ModelSettings", "RouteClassification", "RuntimeModelCatalog"]
