# AGENTS.md

Rules, invariants and traps the code at the edit site does not show: each bullet is the rule, at most a clause of why, and one pointer.

## Project rules skill

Area-specific rules live in the `dev-rules` project skill at `.agents/skills/dev-rules/SKILL.md` (with `.claude/skills` symlinked to `../.agents/skills`). Before changing, planning or reviewing code in a covered area, or operating on what `data/` holds, load `dev-rules` and read the `references/` files its path table maps to your files. Antigravity truncates `AGENTS.md` past 24,000 bytes.

## Commands

```bash
uv sync --all-groups             # once per fresh checkout: plain `uv run` syncs only the dev group
uv run pytest                    # tests, coverage gate: 80% (--cov-fail-under)
uv run pytest <path> --no-cov    # focused run; without --no-cov a subset exits 1 on that gate
uvx pre-commit run -a            # canonical pre-push check (= make fmt)
uvx pre-commit run ty -a         # type check alone; `uvx ty check` rejects --all-groups (a uv flag)
uv run usage-report              # operator tools act on ./data, so run from the repo root
uv run feedback-report
uv run modify-balance
uv run regen-memories
```

An operator tool with a `main()` and arguments lives in `src/discordbot/admin/` and runs as `uv run <command>` through `[project.scripts]`, since an entry point cannot reach `scripts/`, which the wheel does not package. An edit-and-run development script stays in `scripts/develop/` and runs as `uv run python -m scripts.develop.<name>`.

Secrets and API endpoints come from `.env` (see `.env.example`); only `DISCORD_BOT_TOKEN` is required to boot. All non-secret bot defaults (logging, usage records, memory git, media hosting, economy, presence, emojis, reply switches, runtime models, and per-server auto-expand defaults) live in `src/discordbot/config.toml` and are deep-merged with optional `./data/config.toml` (`typings/config.py::load_bot_config`). OpenAI clients are lazy `cached_property`s, so an empty `OPENAI_API_KEY` fails at first use (a turn, or a non-empty memory resume in `on_ready`), not at boot.

Debug from `./data/logs`, which holds logfire output only: a `print()`, a stdlib logger with no logfire handler and whatever nextcord prints to `sys.stderr` never land there. The owner is Discord user `mai9999` (id `1010833712956592200`): match either to pick out his turns in `data/logs`, `data/usage` and `data/memories`. What Discord itself holds (which commands are registered, what a message or embed stored) shows in no log: query its REST API from inside `discordbot-bot-1` with the container's own `DISCORD_BOT_TOKEN`, so the token never reaches a command line. Deploying is in `.github/CONTRIBUTING.md#releases`.

Reuse `tests/conftest.py` and the fakes in `tests/helpers/` before writing a new fake. Unit tests inject fake LLM clients and memory writers and need no API credentials; the Tests workflow intentionally provides none.

The suite reads no `.env`, starts with every variable `.env.example` names cleared, and fails a test that reaches past loopback (`tests/conftest.py`), so a local run is CI's credential-less run; a test that needs a setting sets it with `monkeypatch.setenv`. That guard sees only Python's `socket` module, so a client that connects in C (grpc) needs a network namespace to catch (`bwrap --unshare-net --dev-bind / / uv run pytest`; plain `unshare -n` needs root). A run that stalls near the end is usually pytest-xdist deadlocking (#672): rerun with `-n0`, which has never hung on its own, so a hang there is real. A fake can go green without reaching the branch under test (`MemoryWriterAI.evaluate` returns before any model call on empty notes; a reply's overflow chunks hang off the previous chunk's `replies`, not the message's), so mutate the guarded line once before trusting a new test.

`main` requires no status check, so GitHub refuses auto-merge and lets a merge through mid-CI: wait for every check yourself. After merging a fix, offer to redeploy rather than deploying unasked or leaving it unmentioned; a request to redeploy covers merging on green CI, waiting for the publish run, the deploy, and checking the running image's revision against the merge commit. The release binaries crash at startup and the release workflow stays as the template ships it (#960, closed as not planned): do not raise it.

## Runtime Shape

- `cli.py::DiscordBot` runs on `Intents.all()` minus `members` and `presences`, is feature-agnostic (upgrading only `cleanup.db` in `utils/message_cleanup.py`, while optional cogs upgrade their own databases in `setup()`), and reads neutral presence and emoji defaults from `typings/config.py::load_bot_config` (`src/discordbot/config.toml` + optional `./data/config.toml`).
- **A cog is a directory holding `__init__.py` and `cog.py`** (#403); `_load_cogs_sync` raises on a directory missing either but skips plain files and `_`-prefixed entries, so a stray `cogs/<name>.py` or `cogs/_<name>/` silently never loads. After deleting a cog, remove the leftover `cogs/<name>/` directory in every checkout that ran it (its ignored `__pycache__/` keeps it alive), or it raises at boot.
- **`cogs/<name>/__init__.py` carries no re-exports**, or importing any helper under it runs the whole cog body; nothing but this line enforces it.
- **`services/` sits below the cogs** (#403): Discord-free (`tests/guards/test_package_layering.py::test_services_never_reaches_discord`, transitively) and never importing `cogs/`; `utils/` and `typings/` import from neither.
- **A module goes in `services/` only if a second cog needs it, an engine there needs it, or it is that engine's own vocabulary** (single-caller `services/memory/server_prompts.py` sits beside `prompts.py`, a platform's bare URL pattern in `services/platforms/`); else it stays in its one cog directory. A feature with a Discord surface and a shared engine (economy, memory, each link platform) deliberately spans both.
- **No prefix command or `on_command_*` handler can fire**: no `command_prefix` is passed and `on_message` dispatches nothing, so adding one needs both. A raising slash command is logged only by `DiscordBot.on_application_command_error`, which also tells the caller once the command has answered or deferred, so a command that reports its own failure must catch it rather than re-raise.

## Cog Rules

- **Design a user-facing surface in an artifact before building it.** A panel, view, modal, embed layout or PNG board is settled on a published mockup (Discord-faithful chrome, both themes, real copy, every screen the flow reaches) carrying each open question with a recommendation; wording-only work skips this.
- **Follow `.github/CONTRIBUTING.md#adding-or-removing-a-cog`** for the guard-enforced cog contracts (`INSTALL_CONTEXTS` / `INTERACTION_CONTEXTS`, `LoggedView` / `LoggedModal`, `__init__.py` on nested cog subpackages, and cog-owned migrations in `setup()`).
- **A permission the bot had is not one it has.** Server admins change channel overwrites without notice, so any channel REST call can answer `403 Missing Access`: report it through `utils/discord_errors.py::report_discord_failure` (or `utils/message_cleanup.py::edit_or_report` for tracked public edits) at `warn` with the ids and no traceback (the one carve-out from `.github/CONTRIBUTING.md#logging`) and escalate nothing. A slash command answers on its own token without consulting channel permissions, so a reply landing proves no read access (#694), and anything that returns to such a message later (delete, edit, history read) must go through the interaction while its token is alive.
- **One `commands.Cog` subclass per `cog.py`**.
- **Every user-visible command or behavior change updates `cogs/gen_reply/capabilities.md` in the same change, and a command change also updates the three README command tables.** The answer model knows the bot's features only from that document.
- **Only `InteractionContextType.bot_dm` is a direct message**: `private_channel` also covers group DMs and other people's DMs, and a DM reading hands recall a `dm_partner_id` that opens every compartment the owner has.

## AI Pipeline

### Backend and invariants

- **Every runtime conversation is `AsyncOpenAI` on the Responses API through the LiteLLM proxy (`OPENAI_BASE_URL`).** Never switch back to Chat Completions or import a provider-native SDK (`google-genai`, `anthropic`, `xai-sdk`) into any request path other than the non-proxied ones listed next.
- **The non-proxied paths are the only direct-to-Google calls, and they forgo proxy-side cost/usage tracking**: attachment and linked-post Files API uploads (`cogs/gen_reply/attachment/gemini_file_api.py`, `cogs/gen_reply/files_api.py`), the YouTube answer turn (`cogs/gen_reply/interactions.py`, `cogs/gen_reply/non_streaming.py`), omni video (`VideoGenerator`, VIDEO route and `<generate-video>`), `<generate-music>` (`MusicGenerator`) and deep research (`cogs/research/`).
- **Every direct-to-Google generation call is `interactions.create`; `generate_content` must not come back.** Never read a `generate_content` or proxy result as what `interactions.create` can do, or back, since their capability gates differ (see `.agents/skills/dev-rules/references/responses-api.md`).
- **Runtime model strings live in `[global.models]` (`src/discordbot/config.toml`, exposed through `typings/models.py::RuntimeModelCatalog`); pick a tier by whether the output's shape is already fixed.** Before repointing a tier or widening `RouteClassification.effort`, read `.agents/skills/dev-rules/references/responses-api.md` for effort lookup, Gemini deprecation checks, per-key dispatch, Responses naming, retryability (`utils/llm_errors.py`), and pricing (`utils/model_pricing.py`), and `.agents/skills/dev-rules/references/reply-pipeline.md` for `gen_reply`-only rules (IMAGE/VIDEO aesthetics, kill-switches on `LLMConfig`, inline markers, and attachment tracing).
- **Always read `./data/logs` for LLM failures rather than trusting Discord's user-facing text**: `utils/llm_errors.py::extract_friendly_error` masks operator/backend LLM failures (`401`/`402`/`403`/`404`/`429`/`5xx`, matching codes, and non-timeout connection errors) behind `HIGH_DEMAND_ERROR_NOTICE`, while callers log the raw exception with `_exc_info=exc` first.
- **The owner does not trade quality for inference cost**: never propose a cheaper tier or a tighter memory limit on cost grounds. `slow_model` candidates are the newest snapshot or a pro one; never propose anything else for it on latency grounds.
- **Author runtime prompts in English** (prompts, tool descriptions, `TTS_STYLE_DIRECTIVE`). Chinese stays only as quoted data (a token the context carries, such as `## 成員稱呼`, or a sample of what users type or a note should say), the 轉帳 caution in `COMMON_PROMPT` and the bot's name 破貓.
- **A runtime prompt whose output is posted to Discord as markdown embeds `typings/discord_markdown.py::DISCORD_MARKDOWN_RULES`**; output that stays model input (a memory note) or plain text (a thread title) does not.
- **QA and media persona replies ride `cogs/gen_reply/streaming.py::stream_answer` by default, or `cogs/gen_reply/non_streaming.py` when `[global.reply].streaming_enabled = false` in `config.toml` (both single attempt, leaving retries to SDK defaults and fallbacks to the proxy), and `cogs/gen_reply/generation.py::ImageGenerator` retries an empty result once.**

## Other Cogs

- **`cogs/template/` is a real loaded cog, not a scaffold to copy or delete**. For link-expansion failure classification (`utils/link_errors.py`) and scratch downloads (`utils/scratch_dir.py::scratch_directory`), see `.agents/skills/dev-rules/references/link-expansion.md`.

## Conventions

- **Shared conventions live in `.github/CONTRIBUTING.md`** (`#code-conventions`, `#comments`, `#logging`, `#tests-and-quality-gates`, `#text-formatting`, `#releases`, `#adding-or-removing-a-cog`), including `tests/guards/test_no_bare_star.py`, `tests/guards/test_async_order_contracts.py`, and `tests/guards/test_field_descriptions.py`. Do not repeat or contradict them here.
- **A new env-backed setting updates `.env.example` in the same change**: it is the only settings list a deployer sees and the list the suite clears, and `tests/guards/test_env_example.py` fails on a settings field it does not name (a commented `# NAME=` line counts); a missing one is backfilled with a comment stating its default. The deployment's real value is appended to the repo's `.env` (`>>`), never by reading that file, which holds secrets.
- **A configured host path behind an optional feature degrades, never gets created**: check it at use, fall back to the feature-off path, never `mkdir` it or fail startup over it, and keep its compose mount opt-in (`[global.media_hosting].serve_dir` is the shape). The required `./data` mount fails fast on purpose.
- **Pure shared result types, enums and constants go in `typings/`** when they depend on no cog or util.
- **A model passed as `text_format=` is prompt text**: its class docstring and every `Field(description=...)`, nested models' included, reach the model as the JSON schema's descriptions, so editing one is a behavior change, not a docs fix. Re-derive the set by grepping `text_format=` before any docs sweep (`typings/models.py::RecallRouteClassification` has the measured case).
- **A cap on how much content one model request carries lives in `typings/context_budgets.py`**: one whose binding leaves the model seeing LESS while the request still goes out (the module docstring lists what stays out). No name scan enforces this.
- **`ty` types a pydantic field of a class pydantic cannot model (`nextcord.Message`, `genai.Client`, `AsyncOpenAI`) as `Any` in the synthesized `__init__`**, so a wrong handle handed to a `gen_reply/` service surfaces only at runtime; modelled types stay checked (`int` becomes `LaxInt`). `SkipValidation[T]` does not change the signature, only runtime validation.
- **In a refactor, inline a short, mechanical helper with one caller, but keep one that is independently tested, called in a loop, half of a symmetric pair, names a domain rule or formula, carries a load-bearing docstring, keeps a large function manageable, or is a required interface, callback or scaffold.**
- **A schema change ships as an Alembic migration in its database's own history** (`alembic.ini` has one section per file under `data/database/`). `cli.py` upgrades `cleanup.db` and loaded cogs upgrade `economy`, `games`, `reply`, and `settings` in `setup()` (`migrations/env.py` loads only the active section's metadata lazily, skipping absent optional cogs on `reply`), while `log_msg` only creates `messages.db`: a later revision of that 4.5 GB file, of any kind (the owner turned down an add-column-only limit), runs offline (`.github/CONTRIBUTING.md#releases`), since batch mode rebuilds the whole table for anything but adding a column or an index. Revisions have no `downgrade()`: a rollback restores the backup. `alembic` opens `data/database/` relative to the working directory, so from the repo root on the deployment host it migrates the live files.
- **No factory around a one-line client constructor** (`AsyncOpenAI`, `genai.Client`): the wiring stays at each call site.
- **When the owner says a redundant- or dead-looking structure is kept on purpose, add a one-line why-kept comment in the same change**.
- **An audit of the project also covers the event loop and restarts**: blocking work inside `async def` (sync `requests`, `yt_dlp`, PIL encoding; LLM latency itself is acceptable), and in-memory state a restart loses that belongs in an existing SQLite database. Measure first: wrap blocking work in `asyncio.to_thread` only when a concurrent coroutine at that site would be unblocked; a sequential `await` loop gains nothing, a pipeline is wrapped whole or not at all, and about 100 ms of CPU is not worth a thread.
- **Import the package's own functions by name and call them unqualified**; a module-namespace import of a package module is a last resort with a reason.
- **No `asyncio.Lock` / `Semaphore` / loop-bound registry may outlive one event loop**: each test runs on a fresh loop and raises `is bound to a different event loop`, and building it lazily does not help. Use `utils/asyncio_locks.py`'s loop-local primitives.
- Do not touch the README badge block unless explicitly asked.

## Documentation Split

- **`README.md` is the canonical user-facing README**; `README.zh-CN.md` / `README.zh-TW.md` mirror its structure.
- **The reply pipeline's visual companion is `assets/artifacts/reply-pipeline.html`**, never the behavioral source of truth over `src/`; a change to the pipeline's SHAPE (a route, a trigger condition, an endpoint or SDK a feature dispatches on, a model tier, a kill-switch) updates it in the same change.
- **The bot's Discord UI preview catalog is `assets/artifacts/ui-showcase.html`**, never the behavioral source of truth over `src/`; any change to a user-visible Discord surface (embed, view, button, modal, expansion card, or PNG board) updates it in the same change.
- **Every HTML artifact under `assets/artifacts/` opens with the source-file comment** (`<!-- This file is the source for the published artifact page (not the behavioral source of truth over src/): edit it, republish it over its published page when you can, and never copy that page back over it. Published at <url> -->`, recording `Published at <url>` once published).
- **`README.md`'s "How a reply happens" mermaid, mirrored into both translated READMEs, is a condensed copy**: those three and the artifact move together or not at all.
