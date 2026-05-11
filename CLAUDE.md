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

# Make shortcuts
make fmt           # == pre-commit run -a
make test          # == uv run pytest
make gen-docs      # regenerate the generated docs/ tree from README/CONTRIBUTING/docstrings
make clean         # wipe caches, reports, pycache, then git gc

# Update MapleStory Artale dataset (scrapes artalemaplestory.com into data/maplestory/*.json)
uv run python scripts/artale_data.py
```

The pytest config lives under `[tool.pytest.ini_options]` in `pyproject.toml` — it auto-discovers tests in `tests/`, runs doctests from modules (`--doctest-modules`), collects coverage into `./.github/reports/` and `./.github/coverage_html_report/`, and runs in parallel (`-n=auto`). Asyncio mode is `auto`, so `async def test_*` works without a decorator.

## Architecture

### Bot runtime (`src/discordbot/cli.py`)

`DiscordBot(commands.Bot)` is the entry point. It enables all intents except `members` and `presences` and registers a 1-minute `status_task`. Cog discovery + loading runs **synchronously inside `__init__`** (not in an async hook), so every cog's application commands are registered with the bot **before** the gateway connects:

1. `_load_cogs_sync()` globs `src/discordbot/cogs/*.py` via `pathlib.Path` (skipping any file whose stem starts with `__`) and calls `self.load_extensions(..., stop_at_error=True)`. Helper packages live in sibling `_<cog>/` folders (e.g. `_gen_reply/`, `_maplestory/`) precisely so they are *not* loaded as extensions.
2. On the first `on_ready` — gated by an `_initial_setup_done` flag so reconnect / resume re-fires are no-ops — the bot calls `sync_all_application_commands()` and starts `status_task`.

**Why cog loading runs in `__init__`, not an async hook:** nextcord's `bot.load_extension(...)` is itself sync, but when it sees an `async def setup(bot)` it dispatches the setup via `asyncio.create_task(...)` and returns *without awaiting*. That left `bot.add_cog(...)` un-executed by the time `sync_all_application_commands()` ran, so the first sync iterated over zero application commands and registered nothing with Discord. The fix is two-part and you should not revert either half: every cog's `setup` is **sync** (`def setup(bot): bot.add_cog(...)`), and the bot loads them inside `__init__` before connecting.

`on_command_error` has pre-built embeds for the common `commands.*` exception types; add new cases there rather than catching in cogs.

### Cog conventions

- Cog = a `commands.Cog` subclass + a module-level **sync** `def setup(bot): bot.add_cog(..., override=True)`. The setup must be sync — see "Bot runtime" for the fire-and-forget bug `async def setup` would cause.
- Every cog is free-standing. Cross-cog calls go through the bot instance or through shared typings, never via direct imports of peer cogs.
- Slash commands use nextcord's `@nextcord.slash_command(...)` with `name_localizations` / `description_localizations` for `en-US`, `zh-TW`, `ja` — see `cogs/help.py` and `cogs/maplestory.py` for the pattern. Do not add a new user-facing string without its localizations.
- Helpers that belong to one cog go into a sibling `_<cog>/` package (e.g. `cogs/_gen_reply/prompts.py`). Those paths are deliberately excluded from auto-load.

### AI pipeline (`cogs/gen_reply.py` + `cogs/_gen_reply/prompts.py`)

Every AI call goes through a single `AsyncOpenAI` client built from `LLMConfig` (`base_url=OPENAI_BASE_URL`, `api_key=OPENAI_API_KEY`). **`OPENAI_BASE_URL` points at a [LiteLLM](https://github.com/BerriAI/litellm) proxy** in production, which accepts OpenAI-format requests and dispatches them to the underlying provider named by the `model` string — OpenAI, Azure OpenAI, Google Gemini, Anthropic Claude, DeepSeek, Vertex, etc. **Consequence**: the request path in `src/discordbot/` never imports `google-genai` or `anthropic` directly — both packages still appear in `pyproject.toml` because `scripts/prompt_dev.py` (a prompt-engineering scratchpad, not part of the runtime) uses them. To use a new model in the bot, pass its LiteLLM model name to the same `AsyncOpenAI` client — do NOT reach for provider-native SDKs. Provider-specific features flow through `extra_body` (e.g. `mock_testing_fallbacks`, `fallbacks=[...]`), and per-token pricing is looked up by `discordbot.utils.model_pricing.get_token_rates`, which lazily fetches the same upstream JSON LiteLLM publishes (`model_prices_and_context_window.json`) and stashes a copy at `data/model_prices.json` for offline-restart fallback.

Model assignments live as `@property` methods on `ReplyGeneratorCogs` (`fast_model`, `slow_model`, `image_model`, `video_model`), each returning a `ModelSettings(name=…, effort=…)` instance from `discordbot.typings.models`. Two things to know:

- **Model strings are volatile.** They swap several times a week (see `git log` for `fix: update model settings…`). Read them from `gen_reply.py` rather than memorising them, and update the property bodies — not the call sites — when changing models.
- **`slow_model` is time-of-day dispatched.** During UTC 09:00–17:00 on weekdays (Gemini Pro's known overload window) it returns the cheaper `gemini-3.1-flash-lite-preview` with `effort="high"`; otherwise `gemini-pro-latest` with `effort="high"`. Don't replace this with a static return — the peak-hours fallback is the whole point.

All chat/routing/captioning calls use the **OpenAI Responses API** (`client.responses.create`), not Chat Completions. Streaming results are always named `responses` (object) and iterated as `response` (loop var) — keep this naming.

Flow per `on_message`:

1. **Trigger gate**. In DMs always respond. In guilds respond only if the raw content contains `<@{bot_id}>`. A Discord reply-notification alone does *not* qualify — this prevents the bot from summoning itself when users reply to a Threads embed or video-download result.
2. **Route**. `_route_message` calls the fast model with `ROUTE_PROMPT` via `client.responses.parse(text_format=RouteDecision)` — `RouteDecision` is a Pydantic model in `typings/models.py` that pins the output to one of `IMAGE` / `VIDEO` / `SUMMARY` / `QA`. A `ValidationError` (e.g. safety filter returned no text, so `model_validate_json(None)` blows up) defaults to `QA`.
3. **Dispatch**:
    - `IMAGE` → `client.images.edit` (when image inputs exist) or `client.images.generate`, then a second fast-model pass with `IMAGE_PROMPT` writes a short caption.
    - `VIDEO` → `client.videos.create` + poll until `completed`, then upload the MP4 as a `File`.
    - `SUMMARY` → slow path with `SUMMARY_PROMPT` and `history_limit=100`.
    - `QA` → slow path with `REPLY_PROMPT` and `history_limit=30`.
4. **Slow path** (`_handle_message_reply` → `_handle_streaming`) streams `response.output_text.delta`, appends a footer `-# {model} · ⬆ {in} ⬇ {out} · ${cost:.8f} · {balance:,} (+{total_tokens:,}) 點` whose rates come from `discordbot.utils.model_pricing.get_token_rates`. The trailing `{balance} (+N) 點` segment is the chat-economy reward — `_handle_streaming` awaits `_award_chat_points` (which calls `add_balance` from `cogs/_economy/database.py`) on the `response.completed` event with `total_tokens = input + output`, and uses the returned new balance in the footer. On DB failure `_award_chat_points` logs via `logfire.warn` and returns `None`, so the footer degrades to the older `+{total_tokens:,} 點數` form rather than blocking the reply. It builds Discord messages lazily: the first 30 chars create a `reply`, subsequent chunks `edit` it. If the stream emits any `response.output_text.annotation.added` event (i.e. the model invoked web search / `googleSearch`), a `🌐` reaction is added on top of the `🆗`.
5. **Tools are model-specific** — `ModelSettings.tools` (in `typings/models.py`) returns Gemini's `googleSearch` + `urlContext`, Claude's `web_search_*` + `web_fetch_*`, or OpenAI's `web_search`, dispatched by substring match on `name`. When adding support for a new provider, extend that property.
6. **Progress UX**: `_handle_reaction` manipulates reactions on the **user's** message — never sends an intermediate status message. Expect the sequence 🤔 → 🔀 → 🎨/🎬/📖/❓ → 🆗 (plus 🌐 if web search fired, or ❌ on error). Preserve this; it's the agreed UX.
7. **Attachment ingestion** (`_get_attachment_parts`) is gated by `get_supported_modalities(model_name=slow_model.name)` — attachments whose modality (image / video / audio, with documents collapsed to "image") the slow model doesn't accept are skipped before any LLM call, so a text-only slow model produces zero attachment parts everywhere. The function returns OpenAI SDK typed content parts (`ResponseInputImageParam` or `ResponseInputFileParam`) and filters failed conversions. It routes each surviving attachment by `content_type`: `image/*` → `_image_to_part` (PIL resize to 1568×1568, JPEG re-encode at quality 85, sent as `data:` URI via `input_image`); everything else (`video/*`, `application/pdf`, `text/plain`, `application/json`, …) → `_attachment_to_part` (raw bytes → base64 data URI, sent as `input_file` with `filename`). Stickers and embed images/thumbnails always go through `_image_to_part`; for embeds the `media.discordapp.net` `proxy_url` is preferred over the origin URL since CDN links expire.
8. History / reference / current messages are fetched in parallel via `asyncio.gather`; keep that pattern — the tasks list is built in a `for` loop on its own line, then gathered. Avoid collapsing into a comprehension.

### Economy (`cogs/economy.py` + `cogs/_economy/database.py`)

Persistent point balances back the chat reward, the casino games, the house ledger, and the `/balance` / `/leaderboard` / `/give` / `/house` slash commands.

- **DB**: a separate SQLite file at `data/economy.db` (NOT `messages.db`). The engine is a module-level `AsyncEngine` singleton in `cogs/_economy/database.py:_engine`, built via `create_async_engine("sqlite+aiosqlite:///...")`. **Do not move it onto a per-instance `cached_property`** (same memory-leak lesson `log_msg.py:_sql_engine` captures for its sync engine). Schema is bootstrapped on import via a one-shot `asyncio.run(_create_schema(...))` so the first user-facing call doesn't race against `CREATE TABLE`.
- **Schema**: `UserAccount(user_id PK, name, balance, total_earned, total_spent, updated_at)`. **There is no `guild_id`** — points are intentionally cross-server, so the same Discord user (and the bot's own house-ledger row) has one balance shared across every guild the bot is in. `/leaderboard` is therefore also a global Top 10, not per-guild.
- **API surface** (all native async — every function opens an `AsyncSession(bind=_engine, expire_on_commit=False)` directly, so tests can monkeypatch `_engine` and the swap takes effect immediately. There are no `_*_sync` helpers and no `asyncio.to_thread` shim; if you're tempted to add one, you don't need it):
    - `add_balance(user_id, name, amount)` — chat reward (only `gen_reply.py` pays; Threads / video / template / maplestory deliberately do not)
    - `place_bet(user_id, name, requested_bet)` — atomic upfront wager withdrawal; clamps over-bets to the available balance ("auto all-in") and returns `PlacedBet(amount, balance_after, is_allin)` or `None` when nothing can be wagered
    - `settle_game(user_id, name, delta)` — gross payout crediting after a wager has already been withdrawn; clamps to ≥ 0 (we never let a stale session leave a negative player balance)
    - `house_settle(user_id, name, delta)` — dealer-side settlement that mirrors every player payout (negated). Does **not** clamp at zero — the bot's running balance can and should go negative when the casino has paid out more than it took in. Used only with `bot.user.id` as the key.
    - `transfer(sender_id, sender_name, receiver_id, receiver_name, amount)` — atomic conditional debit + credit inside one transaction; returns `TransferResult(sender_balance, receiver_balance)` on success, and `None` for self-transfer, non-positive amounts, or insufficient funds
    - `top_n(limit, exclude_user_ids=())` — `/leaderboard` data source; `economy.py` always passes `(bot.user.id,)` so the house never crowds out real players
    - `get_account(user_id)` — returns `(name, balance, total_earned, total_spent)` or `None`; used by `/house` to surface gross flows in addition to net
- **Bet flow for games**: bets are withdrawn up-front via `place_bet(...)`, then the round resolves through `cogs/_games/settlement.py:settle_wager(...)`. That helper computes `payout = max(bet + player_delta, 0)`, credits it with `settle_game(delta=payout)`, and mirrors `-player_delta` into `house_settle(...)` for the dealer ledger. A regular win pays `2 * bet`, push pays `bet`, loss pays `0`, and natural Blackjack pays `int(bet * 2.5)`. Withdrawing first via a conditional DB update is what stops a single user firing two `/dice bet:100` commands in parallel from double-spending the same balance. Over-bets are auto-clamped to the player's balance ("auto all-in") inside `place_bet`; only an empty balance rejects the round outright. Both embeds surface a "已自動 all-in" hint when this kicks in so the player notices.

### Games (`cogs/games.py` + `cogs/_games/`)

Slash commands `/dice` and `/blackjack`, each one game-per-invocation against an AI dealer. Players act independently — there's no lobby, no shared table, no need to "start a game"; whoever runs the slash command is in their own session.

- **Pure rules** live in `cogs/_games/dice.py` (`play_dice`, `render_rolls`, infinite-shoe RNG) and `cogs/_games/blackjack.py` (`BlackjackHand`, `hand_value`, `is_blackjack`, `is_bust`, `settle`). These are side-effect-free; tests inject `random.Random(x=seed)` for determinism, while the cog uses `random.SystemRandom()` in production.
- **Natural Blackjack rules**: `is_blackjack(cards)` means exactly two cards totaling 21 (A + 10-value card), not any later 21. Player natural Blackjack settles immediately and pays 1.5×. Dealer natural Blackjack also settles immediately unless the player also has natural Blackjack, in which case it pushes. The final embed includes an "提前結束原因" field when this skips the Hit / Stand flow.
- **Shared wager settlement** lives in `cogs/_games/settlement.py`. `settle_wager(...)` is the common DB path for every finished wager; `settle_blackjack_round(...)` wraps it after calling the pure `settle()` rule function and returns `BlackjackSettlement` plus dealer-prompt detail text. Keep this as the single DB-settlement path for both initial natural Blackjack hands in `games.py`, interactive Blackjack hands in `views.py`, and dice results in `games.py`.
- **Presentation helpers** live in `cogs/_games/presentation.py`. Outcome labels/colors, auto all-in wording, bet field text, and the final settlement footer are centralized there. `settlement_footer(...)` deliberately formats `莊家餘額` as an absolute ledger balance with no leading `+` for positive values; only the player round delta keeps signed formatting.
- **`BlackjackView`** (`cogs/_games/views.py`) drives the `Hit` / `Stand` flow. `interaction_check` restricts the buttons to the original `owner_id`; `on_timeout` auto-stands so a walk-away still settles. `_finalize` is guarded by an `asyncio.Lock` plus `_settled` flag so simultaneous Hit / Stand / timeout paths cannot pay out the same hand twice. It delegates DB settlement to `settle_blackjack_round(...)` and asks `DealerAI.settle` for one closing line.
- **Dealer hint visibility**: the embed hides the dealer's first card and shows the second card while the hand is in progress. `dealer_visible_value(...)` must therefore use the second dealer card when present, so `DealerAI.hint(...)` never receives hidden-card information.
- **Dealer identity is dynamic.** `GamesCogs._dealer_identity()` returns `(bot.user.id, bot.user.display_name)`; the embeds use that name for the field titles ("`{dealer_name} 的牌`", and the dice-result field name) and the house-ledger row. The dealer's banter goes straight into `embed.description` *without* a name prefix — the embed is sent by the bot, so `log_msg.py` already records it under `bot.user.display_name`, and `gen_reply.py` sees it as its own past output via the message sender, no in-content prefix needed.
- **House ledger row.** Every player settlement is mirrored into the bot's own `UserAccount` row through `settle_wager(..., delta=player_delta)`, which calls `house_settle(delta=-player_delta)`. The bot's row is excluded from `/leaderboard` and surfaced separately by `/house`. In final game embeds, `莊家餘額` is this ledger balance, not this-round profit, so positive balances are shown without a leading `+`.
- **Dealer banter** is `cogs/_games/dealer.py:DealerAI`, a thin wrapper around the same `AsyncOpenAI` client + `gemini-flash-latest` (effort=`none`) that `gen_reply.fast_model` uses. Three entry points — `taunt_bet` (game start), `settle` (round end), `hint` (per-Hit during Blackjack) — each falls back to a hard-coded line when the LLM call fails so the round never stalls on AI hiccups. Prompts live in `cogs/_games/prompts.py` (`DEALER_PERSONA`, `DEALER_TAUNT_BET_PROMPT`, `DEALER_SETTLE_PROMPT`, `DEALER_HINT_PROMPT`) and are intentionally fixed strings — there is no `{dealer_name}` placeholder, the dynamic name only flows into the embed.
- **`GamesCogs.dealer`** is a `cached_property` so the `DealerAI` (and the underlying `AsyncOpenAI` client) is built lazily on first command. Same pattern as `ReplyGeneratorCogs.client` and `AutoUnmuteCogs.client` — three independent clients are intentional, not a deduplication target.

### Threads parsing (`cogs/parse_threads.py`)

Listener that watches every `on_message` for a Threads URL (regex on `threads.net` / `threads.com`) and replies with parsed embeds + downloaded videos. Reaction sequence: `🔗` while parsing, `🆗` on success, `⚠️` on oversize / unparsable, `❌` on unhandled error. The cog deliberately pays no points — only `gen_reply.py` is a reward source.

### Video downloader (`cogs/video.py`)

`/download_video` slash command around `discordbot.utils.downloader.VideoDownloader` (yt-dlp). The single `_deliver` helper is shared between the direct download path and the "file too big, retrying at low quality" fallback path; do not duplicate the message-build logic in those branches. The cog deliberately pays no points — only `gen_reply.py` is a reward source.

**Status messages and the file are delivered through different mechanisms on purpose.** Progress / failure text rides on the deferred placeholder via `interaction.edit_original_message(content=...)` — pure text, no file. The final video file goes out as a fresh `interaction.followup.send(content=..., file=...)`, and the placeholder is collapsed to `"✅"`. An earlier implementation tried to push both the file and a text suffix in a single `edit_original_message(content=..., file=...)` call, but Discord drops the `content` field when a multipart file payload is attached — so the text silently vanished. Do not collapse `_deliver` back into `edit_original_message(file=...)`.

### Auto-unmute (`cogs/auto_unmute.py` + `cogs/_auto_unmute/prompts.py`)

When a moderator times out the bot itself, this cog detects the `on_member_update` transition into a future-dated `communication_disabled_until`, identifies the moderator via the audit log, clears the timeout via `member.edit(timeout=None, reason="auto-unmute")`, then posts a single sassy AI reply through its own cached `AsyncOpenAI` client (separate `cached_property` from `ReplyGeneratorCogs`) using `UNMUTE_PROMPT` and `gemini-flash-latest` with `effort="none"`.

- **Reply target**: `_last_active_channel[guild.id]` (updated on every human `on_message`) with a fallback to `guild.system_channel`. Discord's `member_update` audit entry does **not** carry a channel, so this is the only reliable handle.
- **Audit lookup walks 5 entries** because the `member_update` bucket also covers nickname / mute / deafen edits — only the entry whose diff carries `communication_disabled_until` is the right one.
- The `member.edit(timeout=None, …)` call fires `on_member_update` again with `after_until=None`; the early return at the top of the listener prevents an infinite loop.
- `nextcord.Forbidden` on the audit query (missing `view_audit_log`) is logged and swallowed — the AI just gripes at an anonymous moderator instead of pinging.

### Config (`src/discordbot/typings/`)

Each config is a `pydantic_settings.BaseSettings` with `validation_alias=AliasChoices("ENV_NAME")`, so env-var names are explicit. `.env` is auto-loaded via `dotenv.load_dotenv()` at import time.

- `DiscordConfig` (`typings/config.py`) — `DISCORD_BOT_TOKEN` (required), `DISCORD_TEST_SERVER_ID` (optional, enables instant-sync to one guild).
- `LLMConfig` (`typings/llm.py`) — `OPENAI_BASE_URL`, `OPENAI_API_KEY`.
- `ModelSettings` / `RouteDecision` (`typings/models.py`) — not env config but the same package. `ModelSettings(name, effort)` is the unified handle for a model: `name` goes to `client.responses.create(model=…)`, `reasoning` builds the Responses-API reasoning block (`Reasoning(effort=…, summary="auto")`), and `tools` dispatches the right web-search shape per provider. Accepted input modalities are not on `ModelSettings`; callers look them up via `get_supported_modalities(model_name=…)` from `utils/model_pricing.py` to keep `typings/` free of `utils/` imports. `RouteDecision` is the Pydantic schema fed to `client.responses.parse(text_format=…)` for routing.

When adding a new configurable value, keep the `Field(description=..., examples=...)` descriptions populated — Pydantic Field descriptions are load-bearing in this codebase and must not be stripped during refactors.

### Logging (`src/discordbot/__init__.py`)

`setup_logging()` configures `logfire` with `send_to_logfire=False` (local-only) and tees stdout into `./data/logs/<timestamp>.log` via a `_TeeStream` that strips ANSI escape codes from the file copy. `DiscordBot.__init__` attaches a `LogfireLoggingHandler` to the `nextcord.state` logger so framework events flow into the same pipeline. Use `logfire.info(...)` / `logfire.warn(...)` / `logfire.error(..., _exc_info=True)` in new code — avoid stdlib `logging.*` directly.

### Message logging (`cogs/log_msg.py`)

Every loggable `on_message` is persisted through `MessageLogger._save_messages`, which builds a plain dict and UPSERTs it into the canonical `messages` table in `data/messages.db`. The engine is a module-level singleton (`cogs/log_msg.py:_sql_engine`) — do not move `create_engine()` back onto a per-instance `cached_property`, that pattern was the dominant memory leak. SQLite I/O stays off the event loop via `asyncio.to_thread`, and the connection PRAGMAs enable WAL + `busy_timeout`. The old per-channel / per-DM table layout (`channel_*`, `DM_*`) has been fully migrated out and the one-shot migration script + its backup DB have been deleted; do not reintroduce that schema.

**`data/messages.db` records human messages AND this bot's own replies — but NOT third-party bots.** The author filter lives in `LogMessageCog._should_log`: human messages always log; bot messages log only when `author.id == self.bot.user.id`. Other bots in the same guild (MEE6, etc.) are skipped on purpose so the DB tracks just the conversation participants this bot actually engages with. `MessageLogger.log` itself is filter-free, so a future caller (e.g. a one-off backfill) can log any message without re-implementing the gate.

**Streaming bot replies are captured via UPSERT, not append.** `LogMessageCog` listens to both `on_message` and `on_message_edit`; every write goes through the same INSERT keyed by a new `discord_message_id` column with a partial unique index (`WHERE discord_message_id IS NOT NULL`). The `ON CONFLICT (discord_message_id) DO UPDATE` clause refreshes `content`/`attachments`/`stickers` and leaves `created_at` pinned, so the multi-edit streaming flow in `gen_reply.py:_handle_streaming` (initial `reply()` then several `reply.edit(...)` to attach the `usage_footer`) collapses into one row that mirrors the final on-Discord state. Legacy rows that pre-date the change carry NULL `discord_message_id` and are excluded from the unique index, so they sit as historical artifacts and never UPSERT. The migration in `_write_row_sync` runs `ALTER TABLE ... ADD COLUMN discord_message_id TEXT` only when the PRAGMA table_info pre-check shows the column missing — idempotent and safe to ship to fresh / partially-migrated / already-migrated DBs alike.

### Utilities (`src/discordbot/utils/`)

- `model_pricing.py` — lazy LiteLLM price-table fetch + on-disk cache; exposes `get_token_rates()` (used by the streaming footer) and `get_supported_modalities()` (used by `_get_attachment_parts` to gate attachments against the slow model's accepted modalities, defaulting to `{"text", "image"}` for under-populated upstream entries like Claude). Returns `(0.0, 0.0)` for unknown models so the footer shows `$0.00000000` rather than a bogus estimate.
- `images.py` — `get_pil_image` / `get_image_data` / `convert_base64_to_data_uri`. Inlined from the now-removed `autogen.agentchat.contrib.img_utils` dependency. The image-edit path passes `use_b64=False` to get raw `bytes` because `client.images.edit(image=…)` rejects `ImageInputReferenceParam` dicts.
- `downloader.py` — `yt-dlp` wrapper used by `cogs/video.py`. Returns a `DownloadResult` context manager that unlinks the file on exit.
- `threads.py` — Threads URL parser / scraper for `cogs/parse_threads.py`. Normalises `threads.com` → `www.threads.net` and strips query strings.

### Data dir (`data/`)

- `data/logs/` — per-run tee'd logs.
- `data/maplestory/` — Artale JSON dataset consumed by `cogs/maplestory.py` (via `cogs/_maplestory/service.py`).
- `data/downloads/`, `data/threads/` — ephemeral media scratch space; `cogs/video.py` and `cogs/parse_threads.py` clean up after themselves.
- `data/messages.db` — the SQLite message log, stored in one indexed `messages` table (when using the default config).
- `data/economy.db` — the SQLite point-balance store written by `cogs/_economy/database.py`. One row per Discord user, no `guild_id` (cross-server balances).
- `data/model_prices.json` — cached LiteLLM price table fetched lazily by `discordbot.utils.model_pricing`.

## Coding conventions

- **Ruff** is the formatter and the linter. Line length 99, double quotes, `skip-magic-trailing-comma = true`, Google docstring style. Preview rules are on and the rule set is broad (`F E W C90 I N D UP ANN ASYNC S B A C4 …`). Don't silence with blanket `# noqa` — prefer fixing or, if impossible, the narrowest possible `# noqa: <rule>` with a one-line reason.

- **Type checking**: `mypy` (`[tool.mypy]`, with the Pydantic plugin) runs in pre-commit. Any new public function needs real type hints — `Any` is a last resort.

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

- **Docs**: the entire `docs/` tree is generated output and may be absent in a clean worktree. `make gen-docs` recreates it from `README*.md`, `CONTRIBUTING.md`, source docstrings, and script docstrings; don't hand-edit generated files under `docs/`.

- **Commits**: Conventional Commits, English. PR titles are enforced by `semantic-pull-request.yml`.

## CI signals

- `test.yml` — pytest across Python 3.12 and 3.13 on every push/PR. Coverage must stay ≥ 80%.
- `code-quality-check.yml` — `pre-commit run -a` on PRs. Running this locally before pushing is the fastest feedback loop.
- `build_image.yml` / `build_release.yml` — Docker image to `ghcr.io/mai0313/discordbot` on main/tag, plus cross-platform PyInstaller binaries + PyPI publish on tags.
- `code_scan.yml` — GitLeaks, Trufflehog, CodeQL.

## Non-obvious things to remember

- **Do not touch the README badge block.** It may be outdated, but it is curated — leave those `[![...]]` lines alone during refactors.
- **LiteLLM is how multi-provider works.** The `openai` SDK is the only LLM SDK used in `src/discordbot/`; every provider (Gemini / Claude / OpenAI / Azure / DeepSeek / …) is reached by passing its LiteLLM model string to the same `AsyncOpenAI` client. Don't add `google-genai` / `anthropic` imports to request-path code — change the model string and, if needed, stash provider-specific knobs under `extra_body`. (`scripts/prompt_dev.py` is the lone exception: a prompt-engineering scratchpad that uses provider-native SDKs by design.)
- **Prompts live only in `cogs/_gen_reply/prompts.py`.** Service logic and constants stay in `gen_reply.py`; do not mass-extract helpers just for symmetry.
- **The bot intentionally does not send an intermediate "thinking…" message.** Status is always communicated via reactions on the user's message.
- **`AsyncOpenAI` is a `cached_property` on `ReplyGeneratorCogs`** (and separately on `AutoUnmuteCogs`), so the client is constructed lazily on first use — avoid moving it into `__init__` (it would fail at import time when env vars aren't loaded yet in tests). Each cog owns its own client; this is intentional, not a bug to "deduplicate".
- **Gemini quirk**: when `reasoning.effort != "none"`, the OpenAI-compat layer prepends `\n\n\n` to streamed text. `_handle_streaming` strips leading newlines on the first delta (`content_started` flag) to work around this — don't remove that guard.
- **Gemini thought summary only flows through the Responses API.** The Chat Completions path silently drops Gemini's thought trace via LiteLLM. The Responses API path, when `reasoning.effort != "none"` and `reasoning.summary` is set, emits **both** `reasoning_summary_text.delta` (condensed) and `reasoning_text.delta` (full) stream events — handle both if you want to surface reasoning. `_handle_streaming` currently only consumes `response.output_text.delta`; don't swap this call site back to Chat Completions for "simplicity".
- **Responses API role ↔ content-type pairing is strict on current OpenAI models** (older OpenAI models and Gemini/Claude via LiteLLM are lax, which has masked violations in the past — so the same payload can silently work on the slow model and fail after a model swap). Rules:
    - `role=user` / `system` / `developer` → content parts must be `input_text` / `input_image` / `input_file`.
    - `role=assistant` → content parts must be `output_text` / `refusal` only. Hardcoding `{"type": "input_text", ...}` under `role=assistant` raises `Invalid value: 'input_text'`.
    - Build inputs with `EasyInputMessageParam` plus `ResponseInputTextParam` / `ResponseInputImageParam` / `ResponseInputFileParam`, and only `cast("ResponseInputParam", message_list)` at the `client.responses.*` boundary.
    - Prefer the `EasyInputMessageParam` string-content shorthand (`{"role": "...", "content": "plain string"}`) when there are no attachments — the SDK picks the correct part type automatically and preserves assistant-role semantic weighting.
    - Any processed Discord message carrying attachments must fall back to `role=user` (since `output_text` cannot hold `input_image` / `input_file`); `_process_single_message` carries identity via the `{display_name} ({name}) [id: {id}]:` prefix from `_get_cleaned_content`.
    - Separator / header messages (`==== Chat History ====` etc.) use `role=system`, not `developer`, to stay compatible with Gemini/Claude via LiteLLM. The real system prompt is delivered via `instructions`.
- **OpenAI SDK quirk for image edits**: `client.images.edit(image=...)` needs raw `bytes`, not `ImageInputReferenceParam` dicts. `_handle_image_reply` extracts `data_uris` → `get_image_data(use_b64=False)` → `bytes` for this reason.
- **Per-token pricing comes from `discordbot.utils.model_pricing`**, which lazily fetches `model_prices_and_context_window.json` from the upstream LiteLLM repo on first lookup and stashes a copy at `data/model_prices.json` for offline restarts. If a new model name shows `$0.00000000`, the upstream JSON hasn't catalogued it yet — wait for it to land upstream (or temporarily delete `data/model_prices.json` to force a fresh fetch). Do NOT hardcode rates in `gen_reply.py`. The same module also exposes `get_supported_modalities()`, which `_get_attachment_parts` calls directly to decide whether to pass attachments to a given model.
- **`message.snapshots` (Discord's forward feature) is intentionally NOT walked** by `_get_cleaned_content` or `_get_attachment_parts`. Forwards are rare in practice and adding them would double the per-message work; the comments in those methods are the canonical reminder. Revisit if forwarded media becomes common.
- **`BELIEF` (in `_gen_reply/prompts.py`) is currently disabled** in `_handle_message_reply` — the call to inject it as a `context_prompt` is commented out because the model treated it as too prescriptive and refused benign requests. The argument still flows through the function signature for the day someone re-enables it after prompt tuning; don't delete the parameter just because it's unused.
- **`is_dm` short-circuit**: in DMs the bot always responds; in guilds it requires a literal `<@bot_id>` substring in `message.content`. A Discord *reply notification* puts the bot in `message.mentions` without writing the mention into `content`, so the content check is what prevents the bot from summoning itself when a user replies to its own Threads embed or video result.
- **Points are cross-server, on purpose.** `UserAccount` is keyed by `user_id` alone — no `guild_id`. Same Discord user → one balance shared across every guild the bot is in, and `/leaderboard` is global Top 10. If you're tempted to add per-guild scoping, that's a schema change (probably a separate `(user_id, guild_id)` membership table joined into the leaderboard query) — don't try to retrofit it via app-side filters.
- **Bets are withdrawn up-front, payouts credit `bet + delta`.** `settle_wager(...)` owns this conversion for both dice and Blackjack. If you ever change `settle()` in `cogs/_games/blackjack.py` to return absolute payout instead of a delta, the call sites will silently double-pay. Tests in `tests/test_blackjack.py` lock down the delta semantics; tests in `tests/test_economy.py` lock down shared wager settlement and duplicate-finalize protection.
- **The dealer is a real `UserAccount` row, with a deliberately negative balance.** `settle_wager(...)` shadows every player settlement by writing `-player_delta` to the bot's own `user_id` through `house_settle(...)`. Unlike `settle_game`, `house_settle` does *not* clamp at zero — the bot has effectively infinite funds, so its running balance can and should go negative when the casino is losing. `/leaderboard` filters this row out via `top_n(exclude_user_ids=(bot.user.id,))`; `/house` reads it via `get_account` to surface gross flows alongside the (possibly negative) net. Do not format this balance with a leading `+`; positive balances are still balances, not deltas.
- **Only `gen_reply.py` pays points.** `parse_threads.py` and `video.py` previously had per-action rewards but those were intentionally removed; do not re-introduce `add_balance(...)` calls in any cog other than `gen_reply.py` (and `economy.py` / `games.py`, which are the user-facing balance and game-payout flows). `template.py` and `maplestory.py` have always been unpaid.
- **Dealer AI is best-effort.** `DealerAI._ask` returns a hard-coded fallback string on any exception; never let a missing AI line block the round resolution or the DB write. If you add a new dealer entry point, give it a fallback line in `cogs/_games/dealer.py` next to `_FALLBACK_*`.
- **`interaction.edit_original_message(content=…, file=…)` drops `content` when a multipart file payload is attached.** The file uploads fine, but `content` reverts to its previous value. Fix: edit the placeholder with text only, then send the file as a separate `interaction.followup.send(content=…, file=…)`. `cogs/video.py:_deliver` is the canonical pattern; do not regress it back into a single `edit_original_message(file=…)` call.

## Helper skills (use when relevant)

Claude Code has access to these skills for checking **provider-native** behavior when you need authoritative docs or migration guidance. Invoke them as needed — they are reference tools, not required steps. Remember: the runtime path still goes through LiteLLM + `AsyncOpenAI`; do NOT introduce provider-native SDKs into request-path code just because a skill discusses them.

- **`openai-docs`** — official OpenAI API / Responses API / model specs. Use when debugging Responses-API edge cases (role/content pairing, streaming event shapes), confirming behavior of the currently-configured OpenAI model (see the `slow_model` / `fast_model` properties in `cogs/gen_reply.py`), or picking the right OpenAI model string.
- **`gemini-api-dev`** — Gemini model lineup, capabilities, and native API semantics. Use when a LiteLLM-proxied Gemini call behaves oddly (e.g. reasoning/thought-summary quirks, tool parameter shape) and you need the upstream spec to decide whether it's a LiteLLM translation issue or a Gemini-side constraint.
- **`claude-api`** — Anthropic SDK / Claude model specs / migration notes. Use when swapping Claude model strings, tuning Claude-specific knobs surfaced through `extra_body`, or understanding what Claude features (web_search, web_fetch, caching, thinking) LiteLLM actually forwards.
