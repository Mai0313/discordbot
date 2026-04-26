# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All tooling runs through `uv`. No `pip`, no global Python.

```bash
# Run the bot
uv run discordbot                     # same as `python -m discordbot.cli`

# Tests (pytest + asyncio-auto + xdist; enforces --cov-fail-under=80)
uv run pytest                         # all tests
uv run pytest tests/test_download.py  # single file
uv run pytest -k threads              # filter by name
uv run pytest -m "not slow"           # skip tests marked `slow` / `skip_when_ci`

# Lint / format / type-check (bundled in pre-commit)
uv run pre-commit run -a              # the canonical “is this OK to ship” check
uv run ruff check . --fix
uv run ruff format .
uv run mypy src
uv run ty check src                   # Astral's type checker; most rules set to error

# Make shortcuts
make fmt           # == pre-commit run -a
make test          # == uv run pytest
make gen-docs      # regenerate docs/Reference and docs/Scripts via scripts/gen_docs.py
make clean         # wipe caches, reports, pycache, then git gc

# Update MapleStory Artale dataset (scrapes artalemaplestory.com into data/maplestory/*.json)
uv run python scripts/artale_data.py
```

The pytest config lives under `[tool.pytest.ini_options]` in `pyproject.toml` — it auto-discovers tests in `tests/`, runs doctests from modules (`--doctest-modules`), collects coverage into `./.github/reports/` and `./.github/coverage_html_report/`, and runs in parallel (`-n=auto`). Asyncio mode is `auto`, so `async def test_*` works without a decorator.

## Architecture

### Bot runtime (`src/discordbot/cli.py`)

`DiscordBot(commands.Bot)` is the entry point. It enables all intents except `members` and `presences`, registers a 1-minute `status_task`, and on `setup_hook`:

1. Calls `get_cogs_names()` — async-globs `src/discordbot/cogs/*.py` and **skips any file whose stem starts with `__`**. Helper packages live in sibling `_<cog>/` folders (e.g. `_gen_reply/`, `_maplestory/`) precisely so they are *not* loaded as extensions.
2. Loads every discovered cog via `self.load_extensions(..., stop_at_error=True)`.
3. Syncs slash commands — test-guild-only first (when `DISCORD_TEST_SERVER_ID` is set, for instant iteration) then globally.

`on_command_error` has pre-built embeds for the common `commands.*` exception types; add new cases there rather than catching in cogs.

### Cog conventions

- Cog = a `commands.Cog` subclass + a module-level `async def setup(bot): bot.add_cog(..., override=True)`.
- Every cog is free-standing. Cross-cog calls go through the bot instance or through shared typings, never via direct imports of peer cogs.
- Slash commands use nextcord's `@nextcord.slash_command(...)` with `name_localizations` / `description_localizations` for `en-US`, `zh-TW`, `ja` — see `cogs/help.py` and `cogs/maplestory.py` for the pattern. Do not add a new user-facing string without its localizations.
- Helpers that belong to one cog go into a sibling `_<cog>/` package (e.g. `cogs/_gen_reply/prompts.py`). Those paths are deliberately excluded from auto-load.

### AI pipeline (`cogs/gen_reply.py` + `cogs/_gen_reply/prompts.py`)

Every AI call goes through a single `AsyncOpenAI` client built from `LLMConfig` (`base_url=BASE_URL`, `api_key=API_KEY`). **`BASE_URL` points at a [LiteLLM](https://github.com/BerriAI/litellm) proxy** in production, which accepts OpenAI-format requests and dispatches them to the underlying provider named by the `model` string — OpenAI, Azure OpenAI, Google Gemini, Anthropic Claude, DeepSeek, Vertex, etc. **Consequence**: this codebase never imports `google-genai` or `anthropic` at runtime. To use a new model, just pass its LiteLLM model name to the same `AsyncOpenAI` client — do NOT reach for provider-native SDKs. Provider-specific features flow through `extra_body` (e.g. `mock_testing_fallbacks`, `fallbacks=[...]`), and per-token pricing comes from `litellm.model_cost`.

The current model assignments (`DEFAULT_FAST_MODEL`, `DEFAULT_SLOW_MODEL`, `DEFAULT_IMAGE_MODEL`, `DEFAULT_VIDEO_MODEL`) live at the top of `cogs/gen_reply.py` and are swapped frequently — **treat the exact model strings as volatile**, read them from that file rather than remembering them, and update the constants (not the call sites) when changing models.

All chat/routing/captioning calls use the **OpenAI Responses API** (`client.responses.create`), not Chat Completions. Streaming results are always named `responses` (object) and iterated as `response` (loop var) — keep this naming.

Flow per `on_message`:

1. **Trigger gate**. In DMs always respond. In guilds respond only if the raw content contains `<@{bot_id}>`. A Discord reply-notification alone does *not* qualify — this prevents the bot from summoning itself when users reply to a Threads embed or video-download result.
2. **Route**. `_route_message` calls the fast model (`DEFAULT_FAST_MODEL`) with `ROUTE_PROMPT` and classifies the intent as `IMAGE` / `VIDEO` / `SUMMARY` / `QA`.
3. **Dispatch**:
    - `IMAGE` → `client.images.edit` (when attachments exist) or `client.images.generate`, then a second fast-model pass with `IMAGE_PROMPT` writes a short caption.
    - `VIDEO` → `client.videos.create` + poll until `completed`, then upload the MP4 as a `File`.
    - `SUMMARY` → slow path with `SUMMARY_PROMPT` and `history_limit=100`.
    - `QA` → slow path with `REPLY_PROMPT` and `history_limit=30`.
4. **Slow path** (`_handle_message_reply` → `_handle_streaming`) streams `response.output_text.delta`, appends a footer `> **{model}** ⬆ {in} ⬇ {out} ${cost:.8f}` using `litellm.model_cost`. It builds Discord messages lazily: the first 30 chars create a `reply`, subsequent chunks `edit` it.
5. **Tools are model-specific** — `get_tools(model)` returns Gemini's `googleSearch` + `urlContext`, Claude's `web_search_*` + `web_fetch_*`, or OpenAI's `web_search`. When adding support for a new provider, extend this dispatch.
6. **Progress UX**: `_handle_reaction` manipulates reactions on the **user's** message — never sends an intermediate status message. Expect the sequence 🤔 → 🔀 → 🎨/🎬/📖/❓ → 🆗 (or ❌ on error). Preserve this; it's the agreed UX.
7. **Attachment ingestion** (`_get_attachments`) routes each attachment by `content_type`: `image/*` → `_image_to_part` (PIL resize to 1568×1568, JPEG re-encode at quality 85, sent as `data:` URI via `input_image`); everything else (`video/*`, `application/pdf`, `text/plain`, `application/json`, …) → `_file_attachment_to_part` (raw bytes → base64 data URI, sent as `input_file` with `filename`). Stickers and embed images/thumbnails always go through `_image_to_part`; for embeds the `media.discordapp.net` `proxy_url` is preferred over the origin URL since CDN links expire.
8. History / reference / current messages are fetched in parallel via `asyncio.gather`; keep that pattern — the tasks list is built in a `for` loop on its own line, then gathered. Avoid collapsing into a comprehension.

### Config (`src/discordbot/typings/`)

Each config is a `pydantic_settings.BaseSettings` with `validation_alias=AliasChoices("ENV_NAME")`, so env-var names are explicit. `.env` is auto-loaded via `dotenv.load_dotenv()` at import time.

- `DiscordConfig` — `DISCORD_BOT_TOKEN` (required), `DISCORD_TEST_SERVER_ID` (optional, enables instant-sync to one guild).
- `LLMConfig` — `BASE_URL`, `API_KEY`.

When adding a new configurable value, keep the `Field(description=..., examples=...)` descriptions populated — Pydantic Field descriptions are load-bearing in this codebase and must not be stripped during refactors.

### Logging (`src/discordbot/__init__.py`)

`setup_logging()` configures `logfire` with `send_to_logfire=False` (local-only) and tees stdout into `./data/logs/<timestamp>.log` via a `_TeeStream` that strips ANSI escape codes from the file copy. `DiscordBot.__init__` attaches a `LogfireLoggingHandler` to the `nextcord.state` logger so framework events flow into the same pipeline. Use `logfire.info(...)` / `logfire.warn(...)` / `logfire.error(..., _exc_info=True)` in new code — avoid stdlib `logging.*` directly.

### Message logging (`cogs/log_msg.py`)

Every `on_message` is persisted through `MessageLogger._save_messages`, which builds a one-row `pandas.DataFrame` and writes it via `to_sql` into a SQLite DB at `data/messages.db`. The engine is a module-level singleton (`cogs/log_msg.py:_sql_engine`) — do not move `create_engine()` back onto a per-instance `cached_property`, that pattern was the dominant memory leak. The schema is defined implicitly by the `data_dict` fields in this cog.

### Data dir (`data/`)

- `data/logs/` — per-run tee'd logs.
- `data/maplestory/` — Artale JSON dataset consumed by `cogs/maplestory.py` (via `cogs/_maplestory/service.py`).
- `data/downloads/`, `data/threads/` — ephemeral media scratch space; `cogs/video.py` and `cogs/parse_threads.py` clean up after themselves.
- `data/messages.db` — the SQLite message log (when using the default config).

## Coding conventions

- **Ruff** is the formatter and the linter. Line length 99, double quotes, `skip-magic-trailing-comma = true`, Google docstring style. Preview rules are on and the rule set is broad (`F E W C90 I N D UP ANN ASYNC S B A C4 …`). Don't silence with blanket `# noqa` — prefer fixing or, if impossible, the narrowest possible `# noqa: <rule>` with a one-line reason.

- **Type checking**: both `mypy` (`[tool.mypy]`, with the Pydantic plugin) and `ty` (`[tool.ty.rules]`, most rules at `error`) run in pre-commit. Any new public function needs real type hints — `Any` is a last resort.

- **Keyword arguments** are required for **every** function and method call, **including single-argument calls**. Concrete:

    - ✓ `create_engine(url="sqlite:///data/messages.db")` ✗ `create_engine("sqlite:///data/messages.db")`
    - ✓ `re.compile(pattern=r"...")` ✗ `re.compile(r"...")`
    - ✓ `asyncio.create_task(coro=...)` ✗ `asyncio.create_task(...)`
    - ✓ `BytesIO(initial_bytes=data)` ✗ `BytesIO(data)`

    Positional-only calls read as noise in this codebase. There are exactly three exception categories — anything outside them must be named:

    5. **Signature-level positional-only** (Python rejects the kwarg form). Examples: `Path("a/b")` (`*pathsegments`), `RuntimeError("msg")` and other exception constructors (`*args`), `logfire.info("...")` (`msg_template` is positional-only).
    6. **Variadic `*args` collectors**, where each "argument" is a member of a tuple, not a named parameter. Examples: `contextlib.suppress(Exception, OSError)`, `AliasChoices("ENV_NAME")`, `super().__init__(*a, **kw)`.
    7. **One-line builtin idioms** where naming is pure noise: `print(x)`, `len(x)`, `str(x)`, `int(x)`, `s.split(",")`, `s.startswith("/")`. Anything reading like a stdlib idiom — not application logic — qualifies.

    Enforced by code review — no automated lint rule exists for this. When you write or touch a call, the default answer is "name it"; the burden of proof is on the positional form.

- **No intermediate one-level aliases** (`usage = responses.usage` → just use `responses.usage`).

- **`responses` / `response` naming**: whenever you call any LLM SDK (streaming or not), name the return object `responses`; if iterating, the loop variable is `response`. This is enforced by code review.

- **LLM latency matters** in the request path. Don't chain extra LLM calls for cosmetic improvements, and don't add executor/`asyncio.gather` scaffolding for tiny (~100 ms) CPU work without measuring first.

- **Comments**: default to none. Only write a comment when the *why* is non-obvious (hidden constraint, subtle invariant, specific-bug workaround). Do not narrate what well-named code already says, and do not reference tasks / PRs / issues in code comments.

- **Docs**: API reference is auto-generated from docstrings via `scripts/gen_docs.py` into `docs/Reference` and `docs/Scripts`; don't hand-edit those paths.

- **Commits**: Conventional Commits, English. PR titles are enforced by `semantic-pull-request.yml`.

## CI signals

- `test.yml` — pytest across Python 3.12 and 3.13 on every push/PR. Coverage must stay ≥ 80%.
- `code-quality-check.yml` — `pre-commit run -a` on PRs. Running this locally before pushing is the fastest feedback loop.
- `build_image.yml` / `build_release.yml` — Docker image to `ghcr.io/mai0313/discordbot` on main/tag, plus cross-platform PyInstaller binaries + PyPI publish on tags.
- `code_scan.yml` — GitLeaks, Trufflehog, CodeQL.

## Non-obvious things to remember

- **Do not touch the README badge block.** It may be outdated, but it is curated — leave those `[![...]]` lines alone during refactors.
- **LiteLLM is how multi-provider works.** The `openai` SDK is the only LLM SDK used at runtime; every provider (Gemini / Claude / OpenAI / Azure / DeepSeek / …) is reached by passing its LiteLLM model string to the same `AsyncOpenAI` client. Don't add `google-genai` / `anthropic` imports to request-path code — change the model string and, if needed, stash provider-specific knobs under `extra_body`.
- **Prompts live only in `cogs/_gen_reply/prompts.py`.** Service logic and constants stay in `gen_reply.py`; do not mass-extract helpers just for symmetry.
- **The bot intentionally does not send an intermediate "thinking…" message.** Status is always communicated via reactions on the user's message.
- **`AsyncOpenAI` is a `cached_property` on `ReplyGeneratorCogs`**, so the client is constructed lazily on first use — avoid moving it into `__init__` (it would fail at import time when env vars aren't loaded yet in tests).
- **Gemini quirk**: when `reasoning.effort != "none"`, the OpenAI-compat layer prepends `\n\n\n` to streamed text. `_handle_streaming` strips leading newlines on the first delta (`content_started` flag) to work around this — don't remove that guard.
- **Gemini thought summary only flows through the Responses API.** The Chat Completions path silently drops Gemini's thought trace via LiteLLM. The Responses API path, when `reasoning.effort != "none"` and `reasoning.summary` is set, emits **both** `reasoning_summary_text.delta` (condensed) and `reasoning_text.delta` (full) stream events — handle both if you want to surface reasoning. `_handle_streaming` currently only consumes `response.output_text.delta`; don't swap this call site back to Chat Completions for "simplicity".
- **Responses API role ↔ content-type pairing is strict on current OpenAI models** (older OpenAI models and Gemini/Claude via LiteLLM are lax, which has masked violations in the past — so the same payload can silently work on the slow model and fail after a model swap). Rules:
    - `role=user` / `system` / `developer` → content parts must be `input_text` / `input_image` / `input_file`.
    - `role=assistant` → content parts must be `output_text` / `refusal` only. Hardcoding `{"type": "input_text", ...}` under `role=assistant` raises `Invalid value: 'input_text'`.
    - Prefer the `EasyInputMessageParam` string-content shorthand (`{"role": "...", "content": "plain string"}`) when there are no attachments — the SDK picks the correct part type automatically and preserves assistant-role semantic weighting.
    - Bot replies carrying images must fall back to `role=user` (since `output_text` cannot hold `input_image`); `_handle_image_reply` does this and carries identity via the `{display_name} ({name}) [id: {id}]:` prefix from `_get_cleaned_content`.
    - Separator / header messages (`==== Chat History ====` etc.) use `role=system`, not `developer`, to stay compatible with Gemini/Claude via LiteLLM. The real system prompt is delivered via `instructions`.
- **OpenAI SDK quirk for image edits**: `client.images.edit(image=...)` needs raw `bytes`, not `ImageInputReferenceParam` dicts. `_handle_image_reply` extracts `data_uris` → `get_image_data(use_b64=False)` → `bytes` for this reason.
- **`litellm.model_cost`** is the source of truth for per-token pricing in the reply footer. If a new model name shows `$0.00000000`, it's because Litellm hasn't catalogued it yet — update Litellm, don't hardcode rates.

## Helper skills (use when relevant)

Claude Code has access to these skills for checking **provider-native** behavior when you need authoritative docs or migration guidance. Invoke them as needed — they are reference tools, not required steps. Remember: the runtime path still goes through LiteLLM + `AsyncOpenAI`; do NOT introduce provider-native SDKs into request-path code just because a skill discusses them.

- **`openai-docs`** — official OpenAI API / Responses API / model specs. Use when debugging Responses-API edge cases (role/content pairing, streaming event shapes), confirming behavior of the currently-configured OpenAI model (see `DEFAULT_SLOW_MODEL` in `cogs/gen_reply.py`), or picking the right OpenAI model string.
- **`gemini-api-dev`** — Gemini model lineup, capabilities, and native API semantics. Use when a LiteLLM-proxied Gemini call behaves oddly (e.g. reasoning/thought-summary quirks, tool parameter shape) and you need the upstream spec to decide whether it's a LiteLLM translation issue or a Gemini-side constraint.
- **`claude-api`** — Anthropic SDK / Claude model specs / migration notes. Use when swapping Claude model strings, tuning Claude-specific knobs surfaced through `extra_body`, or understanding what Claude features (web_search, web_fetch, caching, thinking) LiteLLM actually forwards.
