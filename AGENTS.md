# AGENTS.md

Rules, invariants and traps the code at the edit site does not show: each bullet is the rule, at most a clause of why, and one pointer.

## Commands

```bash
uv sync --all-groups             # once per fresh checkout: plain `uv run` syncs only the dev group
uv run pytest                    # tests, coverage gate: 80% (--cov-fail-under)
uv run pytest <path> --no-cov    # focused run; without --no-cov a subset exits 1 on that gate
uvx pre-commit run -a            # canonical pre-push check (= make fmt)
uvx pre-commit run ty -a         # type check alone; `uvx ty check` rejects --all-groups (a uv flag)
```

Config comes from `.env` (see `.env.example`); only `DISCORD_BOT_TOKEN` is required to boot. OpenAI clients are lazy `cached_property`s, so an empty `OPENAI_API_KEY` fails at first use (a turn, or a non-empty memory resume in `on_ready`), not at boot.

Debug from `./data/logs`, which holds logfire output only: a `print()`, a stdlib logger with no logfire handler and whatever nextcord prints to `sys.stderr` never land there. What Discord itself holds (which commands are registered, what a message or embed stored) shows in no log: query its REST API from inside `discordbot-bot-1` with the container's own `DISCORD_BOT_TOKEN`, so the token never reaches a command line. Deploying is in `.github/CONTRIBUTING.md#releases`.

Reuse `tests/conftest.py` and the fakes in `tests/helpers/` before writing a new fake. Unit tests inject fake LLM clients and memory writers and need no API credentials; the Tests workflow intentionally provides none.

Locally the tests still read a real `.env`: `load_dotenv` walks up from the package, so a worktree under the checkout reads the parent's. Reproduce CI's credential-less run with the variables set empty (`OPENAI_API_KEY= uv run pytest`), not unset, which dotenv refills. A run that stalls near the end is usually pytest-xdist deadlocking (#672): rerun with `-n0`, which has never hung on its own, so a hang there is real. A fake can go green without reaching the branch under test (`MemoryWriterAI.evaluate` returns before any model call on empty notes; a reply's overflow chunks hang off the previous chunk's `replies`, not the message's), so mutate the guarded line once before trusting a new test.

`main` requires no status check, so GitHub refuses auto-merge and lets a merge through mid-CI: wait for every check yourself. The release binaries crash at startup and the release workflow stays as the template ships it (#960, closed as not planned): do not raise it.

## Runtime Shape

- `cli.py::DiscordBot` runs on `Intents.all()` minus `members` and `presences`.
- **A cog is a directory holding `__init__.py` and `cog.py`** (#403); `_load_cogs_sync` raises on a directory missing either but skips plain files and `_`-prefixed entries, so a stray `cogs/<name>.py` or `cogs/_<name>/` silently never loads. After deleting a cog, remove the leftover `cogs/<name>/` directory in every checkout that ran it (its ignored `__pycache__/` keeps it alive), or it raises at boot.
- **`cogs/<name>/__init__.py` carries no re-exports**, or importing any helper under it runs the whole cog body; nothing but this line enforces it.
- **`services/` sits below the cogs** (#403): Discord-free (`tests/test_package_layering.py::test_services_never_reaches_discord`, transitively) and never importing `cogs/`; `utils/` and `typings/` import from neither.
- **A module goes in `services/` only if a second cog needs it, an engine there needs it, or it is that engine's own vocabulary** (single-caller `services/memory/server_prompts.py` sits beside `prompts.py`, a platform's bare URL pattern in `services/platforms/`); else it stays in its one cog directory. A feature with a Discord surface and a shared engine (economy, memory, each link platform) deliberately spans both.
- **No prefix command or `on_command_*` handler can fire**: no `command_prefix` is passed and `on_message` dispatches nothing, so adding one needs both. A raising slash command is logged only by `DiscordBot.on_application_command_error`, which also tells the caller once the command has answered or deferred, so a command that reports its own failure must catch it rather than re-raise.

## Cog Rules

- **Design a user-facing surface in an artifact before building it.** A panel, view, modal, embed layout or PNG board is settled on a published mockup (Discord-faithful chrome, both themes, real copy, every screen the flow reaches) carrying each open question with a recommendation; wording-only work skips this.
- **A permission the bot had is not one it has.** Server admins change channel overwrites without notice, so any channel REST call can answer `403 Missing Access`: log it at `warn` with the ids and no traceback (the one carve-out from `.github/CONTRIBUTING.md#logging`) and escalate nothing. A slash command answers on its own token without consulting channel permissions, so a reply landing proves no read access (#694), and anything that returns to such a message later (delete, edit, history read) must go through the interaction while its token is alive.
- **A Discord `400001 Access to file uploads has been limited` on a guild days old lifted by itself (#719)**: check the guild's age before touching code.
- **A feature that shares schema, concepts or many helpers with an existing cog extends that cog** (the loan commands live in `cogs/economy/cog.py`); only a low-coupling feature gets a new one. Do not add a status or query command just for symmetry; fold the information into an existing command's embed, as `/balance` does (a preference: `/credit status` and `/central_bank status` exist).
- **One `commands.Cog` subclass per `cog.py`**.
- **A shared slash-option vocabulary is a `Literal`, never a `StrEnum`** (`typings/video.py::VideoQuality` has why and the sites a new preset must reach). `SlashOption(default=...)` is typed `Any` upstream, so `ty` cannot check a default.
- **Every user-visible command or behavior change updates `gen_reply/capabilities.md` in the same change, and a command change also updates the three README command tables.** The answer model knows the bot's features only from that document.
- **Aligned-width embeds use `utils.discord_embeds.embed_spacer_payload(..., target=)`**.
- **Content past embed limits is paginated, rendered to a PNG via `attachment://...`, or clipped with a visible notice (`utils/discord_embeds.py::clip_to_utf16_limit`); never truncate silently.**
- **Markdown headings render reliably only inside `embed.description`.**
- **Never hardcode a Discord attachment-size ceiling; read it through `utils/media_delivery.py::upload_limit_for(guild=...)`, also passing `interaction=` when the media goes out through one**.
- **Decide attach vs host vs drop only in `MediaDeliveryPlanner.plan`; keep the Discord send and the degradation policy at each site**. Pass `envelope_margin=MEDIA_ENVELOPE_MARGIN` for a combined attach or one riding with embeds, and keep every new site host-free under `MEDIA_HOSTING_ENABLED=false` (module docstring).

## AI Pipeline

### Backend and invariants

- **Every runtime conversation is `AsyncOpenAI` on the Responses API through the LiteLLM proxy (`OPENAI_BASE_URL`).** Never switch back to Chat Completions or import a provider-native SDK (`google-genai`, `anthropic`, `xai-sdk`) into any request path other than the non-proxied ones listed next.
- **The non-proxied paths are the only direct-to-Google calls, and they forgo proxy-side cost/usage tracking**: attachment and linked-post Files API uploads (`gen_reply/attachment/gemini_file_api.py`, `gen_reply/files_api.py`), the YouTube answer turn (`gen_reply/interactions.py`), omni video (`VideoGenerator`, VIDEO route and `<generate-video>`), `<generate-music>` (`MusicGenerator`) and deep research (`cogs/research/`).
- **Every direct-to-Google generation call is `interactions.create`; `generate_content` must not come back.** Never read a `generate_content` or proxy result as what `interactions.create` can do, or back, since their capability gates differ (see Responses API Gotchas).
- **Cast an `interactions.create()` result to a Protocol, as `generation.py::_InteractionResult` does.**
- **Runtime model strings live only in `RuntimeModelCatalog` (`typings/models.py`); pick a tier by whether the output's shape is already fixed.**
- **The owner does not trade quality for inference cost**: never propose a cheaper tier or a tighter memory limit on cost grounds. `slow_model` candidates are the newest snapshot or a pro one; never propose anything else for it on latency grounds.
- **Look up which efforts a model accepts before repointing a tier, widening `RouteClassification.effort`, or dispatching an existing tier down a new path; never trust a list in the repo, a comment, or memory.** Read `reasoning.supported_efforts` for the `<provider>/<name>` id at `https://openrouter.ai/api/v1/models`, not `supported_parameters` (it lists no values), and cross-check against the provider's own docs where they publish one: Google's thinking page matched openrouter exactly when last compared and is the only source for a snapshot openrouter lacks.
- **Through the proxy an out-of-set effort is an HTTP 200 from the fallback deployment on a non-streaming call and an error on a streaming one (the streaming fallback skips a 400); direct to Google it is a hard failure (#459).** On a non-streaming call only the response's own `model` field proves which model answered, so a tier can look healthy for months and fail the first time a feature dispatches it direct to Google.
- **One `GEMINI_API_KEY` both uploads Files and backs the proxy's conversational aliases (`~/repo/litellm/configs/gemini.yaml`), so the proxy must never pool a conversational alias across keys.**
- **Pass `ReplyToolkit` (`gen_reply/toolkit.py`) explicitly, never through a `ContextVar`**, so a missed hand-off is a `ty` error.
- **Do not reintroduce per-key dispatch** (#582, removed in #620). A peak-hour 503 is upstream model capacity, not a per-project limit (#583: split evenly across keys, zero 429s), while per-key dispatch doubled Files uploads.
- **Author runtime prompts in English** (prompts, tool descriptions, `TTS_STYLE_DIRECTIVE`). Chinese stays only as quoted data (a token the context carries, such as `## 成員稱呼`, or a sample of what users type or a note should say), the 轉帳 caution in `COMMON_PROMPT` and the bot's name 破貓.
- **IMAGE and VIDEO carry no default aesthetic; never reintroduce a code-side static style suffix**, which cannot honor "unless the user specified".
- **Name a Responses API call's result `responses` and its stream items `response`.**
- **Answer proxy-behavior questions from LiteLLM's source in the running container, not its docs, which lag it.** Use `docker exec litellm-litellm-1 python -c "import importlib.metadata as m; print(m.version('litellm'))"` and `/app/.venv/lib/python3.13/site-packages/litellm`, or shallow-clone upstream into a scratch dir for a version the container lacks.
- **A `Vertex_ai_betaException` in a proxy error is the Gemini Developer API**: every Gemini conversational alias is a `gemini/...` deployment, whose route reuses LiteLLM's Vertex transformation. Read the model group a LiteLLM error names before blaming a tier for it.

### Reply pipeline (`gen_reply` orchestration)

- **`/ask` lives on `ReplyGeneratorCogs` because a cog of its own would have to import across cogs**.
- **Read where a turn happens off the surface (`guild_id`, `is_direct_message`, `surface.py`), not `message.guild`**: a synthesized `/ask` message's `guild` is None even in a server. Only a read that needs the `Guild` object itself (server memory, a permission check) still takes `message.guild`, so it is off or at its DM default on `/ask` (`context.py::read_server_memory` has why).
- **Only `InteractionContextType.bot_dm` is a direct message**: `private_channel` also covers group DMs and other people's DMs, and a DM reading hands recall a `dm_partner_id` that opens every compartment the owner has.
- **On `/ask` the interaction token dies 15 minutes after invocation, and the deliverable and the failure notice 404 together, leaving a thinking state that never resolves** (#619). Both media routes generate inside `TurnSurface.delivery_budget_seconds` (`media_reply.py`); the inline `<generate-video>` marker and a QA stream hanging before its first delta are still not bounded by that budget, and neither fits that same fix (#628, open).
- **One triage call decides the route, the effort and the optional recall picks, and its two shapes are measured (#725)**: `RouteClassification`, plus `RecallRouteClassification` with the server memory and candidate blocks only on a turn that has candidates (`routing.py::RouteClassifier.classify`). Never hand it channel history: it would wait on the Discord fetch and pick link sources out of older messages. The price is that a failed parse, or a transient provider error on the call (#742), now costs every field at once (QA, `high`, no picks); the candidate shape, which carries the server memory, failed none of ~470 measured calls.
- **A reply always reads memory, with no caller-side switch to turn it off**, so a test that runs a turn and must stay off the live store takes `memory_isolated_dir` (#576).
- **Progress is reactions on the user's message, never separate status messages** (`utils/reactions.py::ReactionStatusChain`).
- **Every instruction about the replies' tone or persona (not the TTS voice) goes in `gen_reply/prompts.py::PERSONA_CHOICES`, never a route prompt**. A user's `tone.md` or a server's memory asking for banter is accepted, so a reply that sounds toxic is a memory question before a prompt one.
- **Before cutting a runtime prompt line as cruft, count the failure it forbids in the bot's replies in `data/database/messages.db`**. `DO NOT MENTION YOURSELF IN REPLY` and the persona/tone-note leak line each guard a failure those replies show; `try to provide a straight answer` is a deliberate softening.
- **The streaming answer turn is the only LLM request the bot's own code re-issues mid-turn on an error** (`streaming.py::stream_answer_with_retry` has why).
- **Read retryability structurally, never off the message, through `utils/llm_errors.py::llm_status_code` / `is_retryable_llm_error`**.
- **Best-effort media never raises into the pipeline's outer error path**: a voice / inline-image / inline-music / inline-video failure leaves the primary deliverable plus a ⏱️/⚠️ hint (`TurnSurface.hint`). The 🔁 of an answer-stream retry goes through `TurnSurface.mark`, not `hint`: on the gateway it is an independent reaction that outlives the turn on purpose, as the mark that a delivered reply cost more than one attempt.
- **A kill-switch is an `*_enabled` field on `LLMConfig` (`typings/llm.py`) whose description states its own off-behaviour**.
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
- **A link source that hands the model strangers' discussion runs it through `link_sources/__init__.py::defuse_markers`**, or a quoted tag becomes a real render or a memory write; a Douyin caption and author and a Bilibili title, description and uploader stay undefused under `COMMON_PROMPT`'s never-echo rule, whose measurement `tests/test_prompt_guards.py` owns.
- **The voice / image / music / video markers ride QA only and are deliberately separate from the IMAGE/VIDEO routes** (image-first UX): routing decides which runs, so they never double-fire, and those routes never speak or inline-image.
- **The bot's own music clip is re-ingested into later history on purpose**, unlike `reply.wav` (`input.py::_sources_from_parts`): the `<generate-music>` body is stripped from the reply, so the clip is the song's only trace.

### Attachments (`gen_reply/input.py`)

- **A Discord forward has empty `content` / `embeds` / `attachments`; its payload is in `message.snapshots`**.
- **Do not delete the commented `OpenAIFileUploader` / `AnthropicFileUploader` / `GrokFileUploader` branches or their modules as dead code**: they are scaffolding awaiting a verified reference path (`attachment/select.py::build_attachment_handler`).
- **Every attachment to a Gemini answer goes through the Files API whatever its size, by owner decision** (one path and uniform semantics over the upload latency): do not add a small-files-inline threshold; the size rationale in `gemini_file_api.py` is not why. Only a disabled or keyless Files API inlines (`attachment/select.py`).

### Responses API Gotchas

- **Role and part type pair strictly**: `user` / `system` / `developer` take `input_*` parts, `assistant` only `output_text` or `refusal`.
- **Behavior rules go in `instructions` (or `COMMON_PROMPT`), never an input-side wrapper**: `instructions` has `developer` authority over all of `input` but does not persist across turns.
- **Separator messages use `role=system`, not `developer`**, for Gemini and Claude compatibility through LiteLLM.
- **Run a builtin-tools-only positive control before reading any zero as no grounding**: the Responses bridge reports it ONLY as `url_citation` annotations (`vertexaisearch.../grounding-api-redirect/` urls) and never as `groundingMetadata`, which chat completions and the native API do carry along with `webSearchQueries`.
- **A function tool can share a Gemini request with `googleSearch` / `urlContext`, but each surface needs something different** (measured live 2026-07: proxy on LiteLLM 1.93.0, native both direct and via the byte-transparent `/gemini` pass-through): the proxy needs `include_server_side_tool_invocations: true` flat in `extra_body`, without which `_resolve_search_tool_conflict` silently strips the search tools (Google itself 400s the unflagged mix), and with it grounding holds on Responses (streaming too) and chat completions, pro as on flash. `interactions.create` rejects the flag (`Unknown parameter 'tool_config'`) and needs none for `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, `gemini-3.5-flash` and `gemini-3.6-flash`, while both `*-latest` aliases were refused, so suspect a bare alias wherever a capability gate reads the model string.
- **Re-measure the mix with a prompt nonce plus `{"cache": {"no-cache": true}}` in `extra_body`, never from LiteLLM source alone or from one sample**: the proxy's `cache: true` key ignores `extra_body`, so a flagged request replays an unflagged answer, and the drop warning is deduplicated, so it is no per-request signal.
- **A measurement through the proxy records `responses.model` on every sample and runs at concurrency 4 or less**: at 8, 16 of 24 `gemini-3.1-pro-preview` samples came back from the flash fallback with HTTP 200 (2026-09). Measure a route prompt or schema change by replaying logged turns (text from `messages.db` joined on `discord_message_id`), several runs per variant so run-to-run noise is the baseline, cache bypassed (#725).
- **A function call and a grounded answer never share a turn** (2026-07, N=20 per condition on `slow_model`, #409): the call ends the turn with empty text (145/145; whether a search ran on it is unmeasurable), and the next turn with the tool result grounds (19/20); streaming behaved the same, which #409 does not record.
- **Three measured traps before adding a function tool to a turn that searches**: no runtime call sends `extra_body`, so a tool without the flag silently ends grounding on every QA reply; merely offering it drew a call on pure-search questions 20/20, which restraint wording did not fix (10/14) and a sentence exempting live-external-fact questions did (0/14); and naming a tool in `instructions` without attaching it makes the model hallucinate the call (3/12), ending the turn.
- **Never put a mock-testing key in `extra_body`, whatever its value**: since LiteLLM 1.96 `proxy/route_llm_request.py::raise_if_mock_testing_params_disallowed` 400s on the name alone unless `general_settings.dangerously_allow_mock_testing_request_params` is true. The deployed proxy sets that opt-in (`~/repo/litellm/litellm.yaml`), so a mock key is not refused there: a falsy value does nothing and a truthy one injects a real synthetic failure into production traffic.
- **Pricing comes from `utils/model_pricing.py`'s cached LiteLLM table; never hardcode rates.** Its recovery (`cli.py::price_table_task`) deliberately never refetches a table upstream already served: rate drift is a separate question (#473).

## Economy

- **Economy identity is cross-server: `UserAccount` has no `guild_id`.**
- **`total_earned - total_spent == balance` must hold:** every applied positive delta bumps `total_earned`, every negative one bumps `total_spent`.
- **Money inputs are string `SlashOption`s parsed by `utils.amount_parsing.parse_decimal_amount`;** malformed text gets an ephemeral reply before any mutation. Reuse it for any number that can exceed Discord's int cap.
- **`UserAccount.avatar_url` is a last-seen cache (`utils.avatars.guild_avatar_url`); do not backfill existing URLs.**
- **Central-bank minting is bounded by the pool and the per-borrower ceiling, never by the approver.** Approval is any server administrator (`cogs/economy/views.py::is_guild_admin`), which anyone becomes by creating a server.
- **The pool's collateral is per guild; its debt is the whole bank's.** `guild_participant` is recorded only for a rewarded message's author and a `/central_bank` caller, never a `member:` target.
- **Shared social and settlement events are public embeds with scheduled cleanup; personal state, malformed-amount and permission failures are ephemeral.** A validation failure found after the public defer (`/give`, `/games`) is a public expiring embed.
- **A slash command's first followup after a defer keeps the defer's flag**, so a public settlement after an ephemeral defer goes through `utils/interaction_responses.py::send_expiring_followup_after_private_defer`.
- **Exactly one faucet pays an action reward, `cli.py`'s cooldown-gated per-message reward;** other cogs must not add a second (the system-funded mints that settle a loan or a game are not rewards).
- **`apply_vip_blackjack_bonus` and the loan-rate converters stay in `typings/economy.py`,** so code that only formats or settles a number never imports the ledger's engine module (#610).

## Games

- `lobby.py`'s base views keep `raise NotImplementedError`; do not convert them to `abc.ABC`.
- **No LLM anywhere in the games:** do not reintroduce the removed casino `SystemNarrator` banter, the bot-player `reason` text or any AI bot-player. The dealer is a label, not a Discord identity, and posts no narrator messages.
- **Each Blackjack seat and each Dragon Gate bet settles in one atomic step once it resolves:** validate or clamp bets up front, then settle through the helpers.
- **A human player sits at one started Blackjack round at a time** (`cogs/games/seats.py` has why): a game whose stake is settled only after play goes through it too.
- **Action buttons are presence-based: an invalid control is removed, not disabled.**
- **A settled table goes out through `games/interactions.py::publish_final_table`**, which schedules its deletion; never delete terminal public messages in a cog-local loop.
- **The blackjack bot player (`bot_player.py`, no LLM) may drive the casino ledger negative**: the owner asked for a strong bot, so house losses are not a bug to fix in it.

## Memory

- **The path is the privacy boundary** (#408; `store.py`'s docstring has the layout): do not move memory into SQLite.
- **The model never chooses a fact id or filename**: code mints it from compartment + summary (`facts.py::mint_fact_id`), which is what makes path traversal structurally impossible.
- **Keep the compartment boundary structural, not a prompt rule**: each consolidation call sees only the evidence routed to the compartment it writes (`consolidation.py::_consolidate_locked`). Information flows global -> guild, never guild -> global.
- **Pre-#464 prose `member_alias` rows are not backfilled**: `scripts/regen_memories.py` reaches a server scope only as a full evidence rebuild, and a row heals only when a consolidation delta rewrites it; a bare restatement just re-stamps it (`deltas.py::reconfirm_facts`).
- **Every write to the memory tree goes through `store.py`**: `read_memory_document`'s cache is exact only because each fact write through the store bumps the scope's generation (module docstring), so editing the tree out of process while the bot runs is unsupported.
- **The answer model decides what to remember, inline** (#596): the memory markers (see Inline markers) replaced the extraction pass, so do not reintroduce `MemoryWriterAI.extract` or the `PHASE1_PROMPT` / `SERVER_PHASE1_PROMPT` pair. The one exception is `/memory server catchup` (#1080): it reads a channel nobody addressed the bot in, so `propose_server_notes` stands in for the markers there, and its notes still take the same review (`services/memory/catchup.py` has why it bypasses the turn queue).
- **A forget is evidence, never a delete**: `<forget-memory>` writes a `### forget_request` into `raw.md` and consolidation emits the `delete` in a `deletes_only` call, so a forget can only remove a fact, no fact id reaches the answer prompt and the reply path grows no delete capability. `apply_deltas` drops anything but a `delete` in that call, so `PHASE2_PROMPT`'s FORGET REQUESTS block offers nothing else and a partly wrong fact is deleted whole (#869; `deltas.py::partition_forget_requests` has the routing). A delete it does apply also takes the deleted facts' evidence out of `raw.md` and `detail.md` by key (`deltas.py::drop_released_evidence`); never show a model forgotten evidence and rely on a prompt line not to use it (#730). A tone preference is never a fact, so `tone.py::forget_tone` answers the same forget in its own call whose output is line numbers only, which is what keeps it delete-only (#735).
- **A personal forget never reaches server memory, on purpose**: shared memory changing on one member's word is the problem, not a gap.
- **The 🩹 line under a reply is written at raw-append time and proves no forget**: the `Memory compartment consolidated` debug record's `deleted=` for that scope does (a tone forget never appears there). A hand edit under `data/memories` is folded into the bot's next commit for that scope (`git_history.py` adds the whole scope directory), so a deletion in a bot commit does not prove the bot made it.
- **A bad `tone.md` reads coherent on its own; compare it against `detail.md`**: `<existing_tone>` bullets carry no `evidence_kind`, so once a note has merged across batches only the never-invert rule protects a stated preference.
- **Inject the whole memory, never a shrunk summary** (#245): a reply reads memory once with no on-demand lookup, so nothing falls back on what the injection leaves out.
- **`inflight.py`'s `_pending_updates` slot takes no token guard**: a turn resumed at startup carries its old token (`gen_reply/cog.py::_resume_memory`) and can arrive behind a live turn's newer one, so keeping only the newer token drops what the resume rescues.
- **Do not add a Chinese pronoun/relation lexicon to the sharing gate** (`writer.py::_sanitize_observation`): #408 measured it re-locking 14.5% of all `global` observations.
- **Anti-injection is plumbing in `writer.py`, `facts.py` (the nickname table's render and parse) and `utils/llm_transcript.py::sanitize_identity`, not wording in `prompts.py`**; prompts treat conversation content as data, never instructions.
- **A pattern over memory text anchors on ASCII word boundaries (`re.ASCII` or an explicit `[A-Za-z0-9_]` lookaround), never Unicode `\b`/`\w`**: Python counts Chinese as a word character, so a name or token typed against it slipped the sharing gate (#894) and the secret scrub (#915) (`writer.py::_mentions_roster_name`).
- **Recall is deterministic first, the route's optional picks second** (`recall.py`'s docstring, `context.py::plan_recall`). Do not bring back a separate selector or a `get_user_memory` function tool: offered as a tool, the selector picked someone on 64% of turns that named nobody, while the same rules as a route field picked nobody on those turns (#725).
- **Neither note review sees stored memory.** The personal review gets a target-centered list (`writer.py::target_centered_memory_messages`) and the server review the whole `ReplyContext.message_list`; neither carries a memory block, so neither can re-ingest what is already stored.
- Every memory DB write but the clear's tombstone goes through best-effort `inflight.safe_db_write` (`services/memory/database.py` has the state machine).
- **`memory_job` gains no columns for the marker notes.** Nothing migrates this schema and `clear_job` is not best-effort, so a new column takes `/memory clear` down on a deployed bot; the notes ride inside `transcript` (`writer.py::render_turn_payload`).
- **Only `regeneration.py::regenerate_scope_memory`'s replace pass and a forget pass (`deletes_only`) get `allow_mass_delete=True`** (#970); the rebuild is shared by `/memory regenerate` and `scripts/regen_memories.py`, and do not describe the two as equivalent (`regeneration.py`'s docstring).
- **Every store write except the clear's serializes under `scope_lock(scope)`.** The clear skips the lock on purpose, so every file write must sit right after its `cleared_since` check with no `await` in between or an in-flight update resurrects what was cleared (`pipeline.py::clear_scope_memory`).
- **`scope` is the owner key and `compartment` the visibility axis; they are never the same word.**
- **Every clear goes through `pipeline.py::clear_scope_memory`**. A clear is whole only when every tier its docstring lists is neutralized, so an offline wipe with the bot stopped must also create a tombstone or remove that scope's row, and an online clear must use this protocol because staging writers can still exist.
- **A test reading a DEFERRED turn's `memory_job` row must first drain `inflight._db_tasks`** through `tests/helpers/memory.py::wait_for_persisted_writes`.

## Other Cogs

- **An auto-expansion cog subclasses `utils/expansion_cog.py::ExpansionCog` (or its `ConversationExpansionCog`), whose docstrings list the hooks it supplies**: `tests/test_expansion_contract.py` discovers cogs by that base and pins what one may not take back from the shell, so a cog written beside it escapes every check.
- **An expansion cog and an AI link source are a pair, and matching the existing ones is the default; diverging needs a reason.** A cog's `SOURCE` must be a registered source (`tests/test_expansion_contract.py`, `tests/test_link_source_emojis.py`); the reverse is unchecked, so a new registry source is not done until its cog is built or why not is written down, and whether its parser carries the shared conversation shape is decided (`tests/test_platform_shape.py`).
- **Bilibili (a registry source with no cog) and YouTube (not a source, see `watch_video`) are AI-only on purpose, because of length.** A post on an expanding platform is bounded by what anyone may upload there, while these links can be a four-hour video that expanding would download and post.
- **A platform reader raises `utils/link_errors.py` classes (or parents its own onto them)**: the shell picks the reaction and log severity off the class, so any other exception but a `TimeoutError` paints the failure cross.
- **Every request site classifies its failure, media downloads included, and never by testing a `requests` response for truth** (`Response.__bool__` is `self.ok`, so a 429 or 503 is falsy). `link_fetch_error` / `is_retryable_fetch_failure` are the shared classifiers.
- **A URL `parse_threads` publishes must never name whoever shared the post** (`services/platforms/threads.py::ThreadsDownloader._build_conversation`). Do not reintroduce the `total_size` byte guard (media hosting replaced it), a media-count guard, or the permalink follow-up replies (#645).
- **Build a conversation once and never mutate it**: `target` is a `cached_property`, which pydantic does not invalidate on a `chain` append or reassignment.
- **A new platform reader meets the existing readers' traps, each stated in its module**: the full browser header set, not the User-Agent, unlocks a page (`page_json.py`); comment nesting comes from parent ids, never page order, and a page-wide comment scan is scoped by post id (`facebook.py`); a lookalike non-post URL is refused (`is_<platform>_post_url`); and an expiring signed video URL is never a card's link, which outlives it (`parse_instagram/cog.py::_video_link`).
- **The expansion scrapers are not unified onto yt-dlp**: that was weighed and dropped (no coverage gained, a second failure mode); a new video-only platform reuses `services/platforms/ytdlp.py::VideoDownloader`. Nor is the Douyin share-page reader swapped for a third-party download service (tiktokio.com, weighed 2026-10-11: undocumented API, no author, 720p against the bot's 1080p, terms that forbid automation).
- **Threads card images ride as CDN URLs on purpose** (uploading them, #721, was reverted as too heavy): Discord probes a linked embed image once at send and stores a failed probe as a 0x0 image it never retries, so a card occasionally goes out missing an image (2 of 23 Meta CDN images on 2026-09-26). That is the accepted cost; do not bring the upload back unasked.
- **Scratch downloads go in a per-invocation `utils/scratch_dir.py::scratch_directory`, never under `data/`**: `tests/test_scratch_dir.py` catches only `TemporaryDirectory` / `mkdtemp`, and the module docstring has why.
- **`cogs/template/` is a real loaded cog, not a scaffold to copy or delete**.
- **Usage records (`utils/usage_log.py`, #429) cover slash invocations and AI replies only**: a research run gets no record of its own, since its launch is already a `reply.db` `research` row and the `/deep_research` or `QA` turn that started it is recorded as usual. `scripts/usage_report.py` names only what was used, so a zero-use inventory comes from `gen_reply/capabilities.md`.

## Conventions

- **A new env-backed setting updates `.env.example` in the same change**: it is the only settings list a deployer sees, and nothing checks it is complete; a toggle found missing is backfilled with a comment stating its default. The deployment's real value is appended to the repo's `.env` (`>>`), never by reading that file, which holds secrets.
- **An env-configured host path behind an optional feature degrades, never gets created**: check it at use, fall back to the feature-off path, never `mkdir` it or fail startup over it, and keep its compose mount opt-in (`MEDIA_HOSTING_SERVE_DIR` is the shape). The required `./data` mount fails fast on purpose.
- **Pure shared result types, enums and constants go in `typings/`** when they depend on no cog or util.
- **Structured data is a `pydantic.BaseModel`, service and helper classes included**: no `dataclass`, no loose `object` / `Any`. Every field carries `Field(description=...)` (plus `examples=` where useful), a required one as `Field(..., description=...)`.
- **A model passed as `text_format=` is prompt text**: its class docstring and every `Field(description=...)`, nested models' included, reach the model as the JSON schema's descriptions, so editing one is a behavior change, not a docs fix. Re-derive the set by grepping `text_format=` before any docs sweep (`typings/models.py::RecallRouteClassification` has the measured case).
- **A cap on how much content one model request carries lives in `typings/context_budgets.py`**: one whose binding leaves the model seeing LESS while the request still goes out (the module docstring lists what stays out). No name scan enforces this.
- **Log severity follows the ladder in `.github/CONTRIBUTING.md#logging`**, which also owns exception attachment and broad `except`.
- **No logfire record carries the text of a user's message or prompt, only its size** (`prompt_chars=len(user_prompt)`); model output and community nicknames may appear.
- A `# noqa` names its rule and a reason.
- **`ty` types a pydantic field of a class pydantic cannot model (`nextcord.Message`, `genai.Client`, `AsyncOpenAI`) as `Any` in the synthesized `__init__`**, so a wrong handle handed to a `gen_reply/` service surfaces only at runtime; modelled types stay checked (`int` becomes `LaxInt`). `SkipValidation[T]` does not change the signature, only runtime validation.
- **Call with keyword arguments, single-argument calls included**. Positional idioms allowed: `len(x)`, `str(x)`, `Path("x")`, `s.split(",")`, exception constructors, variadic collectors, `logfire.info("message")`.
- **Never spread a `**dict` into `responses.create` / `.parse` or `interactions.create`**; whichever of `service_tier`, `extra_headers` and `extra_body` a call passes goes as an explicit keyword. A spread defeats overload resolution, so the result types as `Any`.
- **A `gather` over a variable number of awaitables takes a `tasks` list built by a `for` loop on its own lines**, never a comprehension or generator expression; existing ones are not a sweep target.
- **No intermediate one-level alias where the original reads as clearly**, such as `usage = responses.usage`.
- **A constant with one caller lives in that function's body, not as a module-level `_NAME`**, unless anything else (a test included) references it, it is user-visible config, or it is a failure-producing wall-clock bound (`typings/timeouts.py`) or a per-request content cap (`typings/context_budgets.py`). A preference for new code, not a sweep.
- **In a refactor, inline a short, mechanical helper with one caller, but keep one that is independently tested, called in a loop, half of a symmetric pair, names a domain rule or formula, carries a load-bearing docstring, keeps a large function manageable, or is a required interface, callback or scaffold.**
- **No legacy, migration or compatibility code**: no wrapper kept for compatibility, no re-export at an old import path, no read-path tolerance for an old data layout; repoint every caller, and clean changed data once, offline.
- **No factory around a one-line client constructor** (`AsyncOpenAI`, `genai.Client`): the wiring stays at each call site.
- **When the owner says a redundant- or dead-looking structure is kept on purpose, add a one-line why-kept comment in the same change** (`typings/models.py::RuntimeModelCatalog.is_peak` is the shape).
- **An audit of the project also covers the event loop and restarts**: blocking work inside `async def` (sync `requests`, `yt_dlp`, PIL encoding; LLM latency itself is acceptable), and in-memory state a restart loses that belongs in an existing SQLite database, as a new table, never a new column. Measure first: wrap blocking work in `asyncio.to_thread` only when a concurrent coroutine at that site would be unblocked; a sequential `await` loop gains nothing, a pipeline is wrapped whole or not at all, and about 100 ms of CPU is not worth a thread.
- **Import the package's own functions by name and call them unqualified**; a module-namespace import of a package module is a last resort with a reason.
- **No `asyncio.Lock` / `Semaphore` / loop-bound registry may outlive one event loop**: each test runs on a fresh loop and raises `is bound to a different event loop`, and building it lazily does not help. Use `utils/asyncio_locks.py`'s loop-local primitives.
- Do not touch the README badge block unless explicitly asked.

## Documentation Split

- **`README.md` is the canonical user-facing README**; `README.zh-CN.md` / `README.zh-TW.md` mirror its structure.
- **The reply pipeline's visual companion is the Artifact `https://claude.ai/code/artifact/a430046d-dd13-4d85-99ac-dd46b434f243`**, never a source; a change to the pipeline's SHAPE (a route, a trigger condition, an endpoint or SDK a feature dispatches on, a model tier, a kill-switch) updates it in the same change. Update it IN PLACE by passing that URL to the publish, or a SECOND artifact is silently minted; it is private and the URL alone grants nothing.
- **`README.md`'s "How a reply happens" mermaid, mirrored into both translated READMEs, is a condensed copy**: those three and the artifact move together or not at all.
