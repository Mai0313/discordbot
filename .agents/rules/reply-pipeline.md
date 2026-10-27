---
description: "Read before changing, planning or reviewing the reply pipeline or deep research: cogs/gen_reply/, cogs/research/, the LLM, model, research and timeout typings, utils/reactions.py, their tests and test helpers, or the logs under data/logs/."
trigger: model_decision
paths:
  - "src/discordbot/cogs/gen_reply/**"
  - "src/discordbot/cogs/research/**"
  - "src/discordbot/typings/llm.py"
  - "src/discordbot/typings/models.py"
  - "src/discordbot/typings/research.py"
  - "src/discordbot/typings/timeouts.py"
  - "src/discordbot/utils/reactions.py"
  - "tests/cogs/gen_reply/**"
  - "tests/cogs/research/**"
  - "tests/guards/test_model_media_parts.py"
  - "tests/guards/test_prompt_guards.py"
  - "tests/typings/test_llm.py"
  - "tests/typings/test_models.py"
  - "tests/utils/test_reactions.py"
  - "tests/helpers/ask_interaction.py"
  - "tests/helpers/gen_reply.py"
  - "tests/helpers/llm_input.py"
  - "tests/helpers/reply_fakes.py"
  - "tests/helpers/research.py"
  - "tests/helpers/test_llm_input.py"
  - "data/logs/**"
---

## AI Pipeline

### Reply pipeline (`gen_reply` orchestration)

- **`/ask` lives on `ReplyGeneratorCogs` because a cog of its own would have to import across cogs**.
- **Read where a turn happens off the surface (`guild_id`, `is_direct_message`, `surface.py`), not `message.guild`**: a synthesized `/ask` message's `guild` is None even in a server. Only a read that needs the `Guild` object itself (server memory, a permission check) still takes `message.guild`, so it is off or at its DM default on `/ask` (`context.py::read_server_memory` has why).
- **On `/ask` the interaction token dies 15 minutes after invocation, and the deliverable and the failure notice 404 together, leaving a thinking state that never resolves** (#619). So the media routes' generation (`media_reply.py`), the QA answer stream's attempts and the inline media markers (`streaming.py`, #628) run inside `TurnSurface.delivery_budget_seconds`, and a new `/ask` path that can outlast it joins them. The media persona reply stays outside on purpose, having no failure notice to save the margin for; the video persona's Files upload and the memory note's late edit are unbounded too.
- **One triage call decides the route, the effort and the optional recall picks, and its two shapes are measured (#725)**: `RouteClassification`, plus `RecallRouteClassification` with the server memory and candidate blocks only on a turn that has candidates (`routing.py::RouteClassifier.classify`). Never hand it channel history: it would wait on the Discord fetch and pick link sources out of older messages. The price is that a failed parse, or a transient provider error on the call (#742), now costs every field at once (QA, `high`, no picks); the candidate shape, which carries the server memory, failed none of ~470 measured calls.
- **A reply always reads memory, with no caller-side switch to turn it off**, so a test that runs a turn and must stay off the live store takes `memory_isolated_dir` (#576).
- **Progress is reactions on the user's message, never separate status messages** (`utils/reactions.py::ReactionStatusChain`).
- **Every instruction about the replies' tone or persona (not the TTS voice) goes in `gen_reply/prompts.py::PERSONA_CHOICES`, never a route prompt**. A user's `tone.md` or a server's memory asking for banter is accepted, so a reply that sounds toxic is a memory question before a prompt one.
- **Before cutting a runtime prompt line as cruft, count the failure it forbids in the bot's replies in `data/database/messages.db`**. `DO NOT MENTION YOURSELF IN REPLY` and the persona/tone-note leak line each guard a failure those replies show; `try to provide a straight answer` is a deliberate softening.
- **Best-effort media never raises into the pipeline's outer error path**: a voice / inline-image / inline-music / inline-video failure leaves the primary deliverable plus a ⏱️/⚠️ hint (`TurnSurface.hint`). The 🔁 of an answer-stream retry goes through `TurnSurface.mark`, not `hint`: on the gateway it is an independent reaction that outlives the turn on purpose, as the mark that a delivered reply cost more than one attempt.
- **`file_api_enabled` gates no feature but the mechanism under several**: it exists for a provider outage where the upload succeeds and the model then refuses the file, which costs the WHOLE reply since the answer carries the failing part (#500: 2026-08-14, every model but `gemini-3-flash-preview`). The VIDEO route's source-clip upload ignores it on purpose: its description scopes it to the answer model.
- **Every turn record carries `message_id` wherever the turn's message is in hand, spans included, and every span is prefixed `gen_reply `** (#568): the console exporter teed to `./data/logs` prints a span only when it STARTS, never a completion line or duration, so an unanchored span drops out of the turn. Code handed no message (`generation.py`, `link_sources/*`) deliberately carries no turn id, being bracketed within milliseconds by anchored records.
- **Attachment records cannot be joined to a turn, and that is unsolved**: `attachment/*` and `files_api.py` serve every turn concurrently, so two replies' upload records interleave with only `filename` / `cache_key` (`name` in `files_api.py`) to tell them apart. To find why an attachment never reached the answer, start from `input.py`'s `gen_reply attachment render` pair for that source message.

### Routes

- **YouTube is watched only on the native Interactions API, by `RouteClassification.watch_video` swapping the one answer turn's backend, never by a second pass** (`gen_reply/interactions.py` has why; `AnswerTurn.stream_answer` holds the gates and logs which one refused).
- **Deep research (`cogs/research/`) goes direct to Google** because, spike-verified, the proxy's interactions transform is not a pass-through (it dropped `agent_config`); Antigravity `background=True` works despite the docs.
- **Do not reintroduce a deep-research tier or a button under the report** (#447).
- **A research report follows the request's language; its status words stay English.**
- **Linked-post context** is one registry of `LinkContextSource` entries, whose docstring holds the add-a-source recipe (`gen_reply/link_sources/registry.py`).
    - **A failed read injects the source's own notice, never an invented body**, so the model never says "I cannot open this link" (the reverted #294's failure), and a page's comment preload is stated as a fraction, never as "the comments".
    - **A Douyin photo gallery can reach the model partially under the attached separator.** Douyin and Bilibili switch their separator on `if media_parts`, which is whole-or-nothing for a single clip but not for a gallery capped at `MAX_DOUYIN_INGEST_IMAGES`, uploaded per image with no count in the text; Threads names what is missing instead (`link_sources/threads.py::PostMedia`).

### Inline markers (`markers.py`, `generation.py`)

- **The answer model picks what to speak and attach inside its own output** (`markers.py` has which tags stay visible): do not move that choice to a post-hoc classifier.
- **A memory marker never chooses whose memory it writes**: `<write-memory>` / `<forget-memory>` (the person replied to) and `<write-server-memory>` (the community) take their scope from the message in `answer.py::_schedule_memory_updates`.
- **A link source that hands the model strangers' discussion runs it through `link_sources/__init__.py::defuse_markers`**, or a quoted tag becomes a real render or a memory write; a Douyin caption and author and a Bilibili title, description and uploader stay undefused under `COMMON_PROMPT`'s never-echo rule, whose measurement `tests/guards/test_prompt_guards.py` owns.
- **The voice / image / music / video markers ride QA only and are deliberately separate from the IMAGE/VIDEO routes** (image-first UX): routing decides which runs, so they never double-fire, and those routes never speak or inline-image.
- **The bot's own music clip is re-ingested into later history on purpose**, unlike `reply.wav` (`input.py::_sources_from_parts`): the `<generate-music>` body is stripped from the reply, so the clip is the song's only trace.

### Attachments (`gen_reply/input.py`)

- **Do not delete the commented `OpenAIFileUploader` / `AnthropicFileUploader` / `GrokFileUploader` branches or their modules as dead code**: they are scaffolding awaiting a verified reference path (`attachment/select.py::build_attachment_handler`).
- **Every attachment to a Gemini answer goes through the Files API whatever its size, by owner decision** (one path and uniform semantics over the upload latency): do not add a small-files-inline threshold; the size rationale in `gemini_file_api.py` is not why. Only a disabled or keyless Files API inlines (`attachment/select.py`).
