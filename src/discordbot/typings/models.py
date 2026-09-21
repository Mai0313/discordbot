"""The model tiers every runtime call dispatches on, and the shapes a triage call fills in."""

from typing import Literal, cast
from datetime import UTC, datetime

from pydantic import Field, BaseModel, computed_field
from openai.types.responses.tool_param import ToolParam
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.reasoning import Reasoning


class ModelSettings(BaseModel):
    """Model name and reasoning effort that should be used together."""

    name: str = Field(
        ...,
        description="LiteLLM model string dispatched on the Responses API.",
        examples=["gemini-3.8-flash", "gemini-3.1-flash-image"],
    )
    # Which efforts a model accepts is per-model and NOT derivable from its name, its family, or
    # a sibling snapshot. The vocabulary itself differs by provider, so a guess is not even wrong
    # in a consistent direction: Gemini has `minimal` and no `xhigh`, OpenAI has `xhigh` and on
    # some models `none`, Anthropic has `max`. Look the model up in openrouter's list
    # (https://openrouter.ai/api/v1/models) under its `<provider>/<name>` id, where `reasoning`
    # carries `supported_efforts`, `default_effort` and `mandatory` (whether thinking can be
    # switched off at all). Sending an effort outside a model's own set is a hard failure rather
    # than a downgrade, so do that lookup instead of trusting a list written down in this tree.
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
        """Responses API reasoning options for this model.

        Returns:
            Reasoning options using this model's configured effort and an
            automatic reasoning summary.
        """
        return Reasoning(effort=self.effort, summary="auto")

    @property
    def tools(self) -> list[ToolParam]:
        """Built-in tool payloads for this model's provider.

        Code execution is intentionally omitted: Gemini and Claude validate every
        uploaded file part against code execution's narrow MIME allowlist and 400 the
        whole request on video / audio / GIF-as-video attachments, so it cannot coexist
        with the attachment ingestion path. Search / url grounding have no such limit.

        Returns:
            Gemini models receive googleSearch and urlContext tools. Claude models
            receive web_search and web_fetch tools. Grok models receive web_search and
            x_search. Every other model receives the OpenAI web_search tool.
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
    """Runtime model settings used by Discord bot LLM paths."""

    @computed_field
    @property
    def is_peak(self) -> bool:
        """Whether runtime model selection is in the peak-hour window.

        No tier reads this today; `slow_model`'s branch on it is parked (see there).

        Returns:
            True during UTC weekdays from 08:00 up to (but excluding) 17:00, otherwise False.
        """
        now = datetime.now(UTC)
        return now.weekday() < 5 and 8 <= now.hour < 17

    @property
    def image_model(self) -> ModelSettings:
        """The model settings for image generation and editing.

        Returns:
            Model settings used with `images.generate` and `images.edit`.
        """
        return ModelSettings(name="gemini-3.1-flash-image")

    @property
    def video_model(self) -> ModelSettings:
        """The model settings for video generation.

        Returns:
            Model settings used with the native Gemini Interactions API (`interactions.create`,
            a bare model name with no provider prefix, since the call goes direct to Google not
            via the proxy). omni unifies text/image/reference/edit video generation, so the same
            model backs plain generation and true source-video editing (`task="edit"`).
        """
        return ModelSettings(name="gemini-omni-1.1-flash")

    @property
    def music_model(self) -> ModelSettings:
        """The model settings for music generation.

        Returns:
            Model settings used with the native Gemini (Lyria) Interactions API (a bare model
            name, no provider prefix, since the call goes direct to Google not via the proxy).
        """
        return ModelSettings(name="lyria-3-clip-preview")

    @property
    def tts_model(self) -> ModelSettings:
        """The model settings for spoken-reply text-to-speech.

        Returns:
            Model settings whose name is dispatched on the `audio.speech` endpoint. Only
            the reply's `<generate-voice>` segments are synthesized, concatenated into one
            clip; the rest of the reply is never spoken. `effort` is unused for TTS.
        """
        return ModelSettings(name="gemini-3.1-flash-tts-preview")

    @property
    def antigravity_model(self) -> ModelSettings:
        """The deep-research agent: a one-shot Antigravity managed agent.

        The only research tier there is.

        Returns:
            The Antigravity managed-agent string dispatched on the Gemini Interactions API
            (direct, not the proxy). `effort` / `tools` are unused on the agent path: the
            agent runs its own internal tool loop.
        """
        return ModelSettings(name="antigravity-preview-05-2026")

    @property
    def triage_model(self) -> ModelSettings:
        """The model settings for filling a shape the caller has already fixed.

        Every call here lands in a slot with no room in it and decides nothing about what to
        write, which is both the seam against `fast_model` and why `minimal` is enough.

        Returns:
            Flash-lite at `minimal`, which is this snapshot's own default effort. Confirm any
            repointed name still lists `minimal` before carrying the effort across (see
            `ModelSettings.effort`).
        """
        return ModelSettings(name="gemini-3.5-flash-lite", effort="minimal")

    @property
    def fast_model(self) -> ModelSettings:
        """The model settings for prose the model composes itself, short of the answer.

        A call here decides what to say rather than how briefly to say it, which is the
        thinking `triage_model` does without. Nothing it produces is the deliverable, which is
        what keeps this tier below `slow_model`.

        Returns:
            Flash at `medium`, two snapshots back from `gemini-3.8-flash`, popular enough now
            to queue behind its own load (observed 2026-08-20).
        """
        return ModelSettings(name="gemini-3.6-flash", effort="medium")

    @property
    def slow_model(self) -> ModelSettings:
        """The model settings for full text replies and strategic reasoning.

        The NAME is read as well as dispatched — the attachment modality gate, the choice of
        attachment renderer and every link source's media ingest all branch on it — so
        repointing this tier changes what reaches the model, not just how well it reasons.

        Returns:
            `gemini-3.1-pro-preview` at `high`, on every hour. The peak-hour split below is
            parked, not deleted: see the comment there.
        """
        # Both branches are pinned to explicit snapshots and never a `*-latest` alias. This is the
        # one tier whose effort is replaced at runtime by the route's grade, and the YouTube
        # answer turn hands that effort straight to the Interactions API as a `thinking_level`
        # (`gen_reply/interactions.py`), where a level the model does not list is a hard failure
        # that loses the whole reply (#459). An alias is the worse end of that: it moves under the
        # deployment, so the set it accepts can change without this file changing. Look the
        # snapshot up (see `ModelSettings.effort`) before repointing either branch, and check its
        # set covers every value `EffortGrade` can emit — including across a provider change,
        # where the effort vocabulary itself differs and a value can stop existing.
        #
        # Through the proxy that rejection does not surface as an error. LiteLLM forwards the
        # level for the model itself to refuse and then answers from the fallback deployment, so
        # the caller sees an HTTP 200 whose `model` field names a different model. A status code
        # proves nothing here; only the response's own `model` does.
        #
        # `gemini-3.8-flash` accepts low / medium / high and NOT `minimal` (Google's thinking
        # table, read 2026-09-02). openrouter does not list that snapshot, so this is the only
        # way to complete the lookup the rule above asks for.
        #
        # The peak-hour branch below is parked rather than deleted: `gemini-3.8-flash` runs into
        # high-demand refusals often enough inside the window that it costs more replies than
        # Pro's queueing did. Uncommenting brings the flash branch's own costs back with it:
        # LiteLLM's price table has no entry for that snapshot, so a peak-hour reply prices at
        # `$0.00000000` in the footer while the attachment modality gate falls back to its
        # `{"text", "image"}` baseline. That gate feeds BOTH renders, so an audio or video
        # attachment does not merely go unuploaded inside the window: its `[attachment: video]`
        # marker never reaches the route or the effort grade either, and the answer model is not
        # told the file existed. A clip posted with one line of text is then answered as if the
        # line were the whole message, which is a wrong answer rather than a degraded one.
        # if self.is_peak:
        #     return ModelSettings(name="gemini-3.8-flash", effort="high")
        return ModelSettings(name="gemini-3.1-pro-preview", effort="high")

    @property
    def memory_writer_model(self) -> ModelSettings:
        """The model settings for everything deciding what reaches long-term memory.

        Returns:
            Model settings for the background memory write calls. The note evaluator and the
            consolidator share one tier because both are gates on the same side: the evaluator
            is the last LLM check before an observation is staged, authoring its fields,
            tightening `sharing` and `durability` (downgrade-only), deduping by
            `normalized_key` and stripping personal-attack wording, and the consolidator turns
            that staging into the fact files plus the tone note. A weaker model on either loses
            memory or leaks it, rather than just failing to record it.
        """
        return ModelSettings(name="gemini-3.1-pro-preview", effort="high")


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
    link_context_sources: list[
        Literal["threads", "facebook", "instagram", "twitter", "douyin", "bilibili"]
    ] = Field(
        default_factory=list,
        description=(
            "Registered linked-post sources whose actual content the user wants analyzed, "
            "summarized, or discussed. Empty when every matching link is incidental. Consumed "
            "only on the QA route to decide which source builders may start."
        ),
    )


class EffortGrade(BaseModel):
    """Structured answer-effort grade returned by the effort model.

    Graded by a call that runs in parallel with the route; the answer model's effort is
    overridden with it on the QA path.

    Deliberately binary, with `high` as the grade an ordinary message gets and `low` as the
    exception that has to be earned: the grader reads text-only parts, so it never sees an
    attachment's content, a linked post, or the history behind a short message, and every one
    of those blind spots hides work rather than inventing it. Both values still have to exist
    on whatever `slow_model` names, and no effort is universal — not even `high` — so
    repointing that tier or widening this grade is checked against the snapshot first (see
    `ModelSettings.effort`).
    """

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


__all__ = ["EffortGrade", "ModelSettings", "RouteClassification", "RuntimeModelCatalog"]
