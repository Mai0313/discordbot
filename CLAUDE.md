# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All tooling runs through `uv`.

```bash
uv run discordbot                # run the bot
uv run pytest                    # tests (coverage gate: 80%)
uv run pre-commit run -a         # the canonical pre-push check
make fmt                         # == pre-commit run -a
make gen-docs                    # regenerate docs/ from sources
```

## Architecture

### Bot runtime (`src/discordbot/cli.py`)

`DiscordBot(commands.Bot)` enables all intents except `members` and `presences`. Cog loading runs **synchronously inside `__init__`** before the gateway connects, so application commands are registered before the first sync:

1. `_load_cogs_sync()` globs `src/discordbot/cogs/*.py` (skipping `__*`) and calls `load_extensions(stop_at_error=True)`. Helper packages live in sibling `_<cog>/` folders so they're not auto-loaded.
2. First `on_ready` (gated by `_initial_setup_done`) calls `sync_all_application_commands()` and starts the 1-minute `status_task`.

**Every cog's `setup` must be sync**, not `async def setup`. nextcord's `load_extension` fires `async def setup` via `asyncio.create_task` without awaiting, leaving cogs un-registered when the first command sync runs. Do not revert this.

`on_command_error` has pre-built embeds for common `commands.*` exception types; add new cases there rather than catching in cogs.

### Cog conventions

- Cog = `commands.Cog` subclass + module-level **sync** `def setup(bot): bot.add_cog(..., override=True)`.
- Cogs don't import peers directly; cross-cog calls go through the bot instance or shared typings.
- Slash commands need `name_localizations` / `description_localizations` for `en-US`, `zh-TW`, `ja`. See `cogs/help.py` and `cogs/maplestory.py` for the pattern.
- Every user-facing feature or behavior change must update `cogs/help.py` (`_HELP_CONTENT`) in the same change so `/help` stays accurate; keep `tests/test_help.py` passing.
- Cog-private helpers live in sibling `_<cog>/` packages (e.g. `_gen_reply/`, `_maplestory/`).

### AI pipeline (`cogs/gen_reply.py` + `cogs/_gen_reply/prompts.py`)

All LLM calls go through one `AsyncOpenAI` client built from `LLMConfig`. **`OPENAI_BASE_URL` points at a [LiteLLM](https://github.com/BerriAI/litellm) proxy** — every provider (OpenAI, Gemini, Claude, DeepSeek, Vertex, …) is dispatched by the LiteLLM `model` string. **Do not import `google-genai` / `anthropic` into the request path**; provider-specific knobs go through `extra_body` instead. (`scripts/prompt_dev.py` is the only place provider-native SDKs are used.)

Models are `@property` methods on `ReplyGeneratorCogs` (`fast_model`, `slow_model`, `image_model`, `video_model`) returning `ModelSettings(name=…, effort=…)`. **Model strings swap frequently** — read them from these properties rather than memorising them, and update the property bodies, not the call sites. **`slow_model` is time-of-day dispatched**: during a known peak window it falls back to a cheaper model; otherwise the regular one. Don't replace this with a static return — the peak-hours fallback is the whole point.

All chat/routing/captioning use the **OpenAI Responses API** (`client.responses.create`), not Chat Completions. Streaming results are always named `responses` and iterated as `response` — keep this naming.

Flow per `on_message`:

1. **Trigger gate**: DMs always respond. Guilds respond only if `<@{bot_id}>` is in `message.content`. A Discord *reply notification* alone doesn't qualify — this prevents self-summoning when users reply to the bot's own embeds.
2. **Route**: `_route_message` calls the fast model via `client.responses.parse(text_format=RouteDecision)` to pin output to `IMAGE` / `VIDEO` / `SUMMARY` / `QA`. `ValidationError` (e.g. safety filter blanked the output) defaults to `QA`.
3. **Dispatch**:
    - `IMAGE` → `client.images.edit` if image inputs exist, else `client.images.generate`; then a fast-model caption pass.
    - `VIDEO` → `client.videos.create` + poll until `completed`, upload MP4.
    - `SUMMARY` → slow path, `SUMMARY_PROMPT`, `history_limit=100`.
    - `QA` → slow path, `REPLY_PROMPT`, `history_limit=30`.
4. **Slow path** (`_handle_message_reply` → `_handle_streaming`): streams `response.output_text.delta` and appends a footer with model name, token usage, cost, and chat-reward balance. The balance comes from `_award_chat_points` (calls `credit_with_repayment` in `cogs/_economy/database.py`) on `response.completed`; on DB failure the footer degrades gracefully rather than blocking the reply. The first 30 chars create a `reply`; subsequent chunks `edit` it. A `🌐` reaction is added if any `response.output_text.annotation.added` event fires (web search was invoked).
5. **Per-provider tools**: `ModelSettings.tools` dispatches by substring match on `name` to Gemini's `googleSearch`+`urlContext`, Claude's `web_search_*`+`web_fetch_*`, or OpenAI's `web_search`. Extend that property for new providers.
6. **Progress UX**: status is communicated via reactions on the **user's** message (🤔 → 🔀 → 🎨/🎬/📖/❓ → 🆗, plus 🌐 / ❌). The bot never sends an intermediate "thinking…" message — preserve this.
7. **Attachment ingestion** (`_get_attachment_parts`) is gated by `get_supported_modalities(model_name=slow_model.name)` from `utils/model_pricing.py`. Attachments whose modality the slow model doesn't accept are dropped before any LLM call. Images go through `_image_to_part` (PIL resize + JPEG re-encode, sent as `input_image` data URI); everything else (`video/*`, `application/pdf`, `text/plain`, …) goes through `_attachment_to_part` as `input_file`. For Discord embeds, `media.discordapp.net` `proxy_url` is preferred over the origin URL since CDN links expire.
8. History / reference / current messages are fetched in parallel via `asyncio.gather`. The tasks list is built in a `for` loop on its own line, then gathered — don't collapse into a comprehension.

### Economy (`cogs/economy.py` + `cogs/_economy/database.py` + `typings/economy.py`)

Persistent point balances backing the global message reward, AI chat reward, casino games, house ledger, loans, and `/balance` / `/leaderboard` / `/debt_leaderboard` / `/give` / `/house` / `/borrow` / `/repay`.

- **DB**: SQLite at `data/economy.db` (NOT `messages.db`). The `AsyncEngine` is a module-level singleton (`_engine`). Do not move it onto a `cached_property` — that's the same memory-leak pattern `log_msg.py` already learned.
- **Schema**:
    - `UserAccount(user_id PK, name, avatar_url, balance, total_earned, total_spent, updated_at, loan_principal, loan_interest, loan_total_borrowed, loan_total_repaid, loan_last_accrual_at, loan_opened_at)`. **No `guild_id`** — points are cross-server by design, so `/leaderboard` is a global Top 10 and loans are also cross-server. `name` and `avatar_url` are last-seen Discord profile metadata refreshed opportunistically on economy writes. Loan columns live on the same row so message reward / chat reward / casino payout can pay debt inside one UPDATE.
    - `PointTransaction(id PK, user_id, kind, delta, balance_after, debt_after, note, occurred_at)` — append-only audit log; every balance-mutating helper writes one row via `_log_transaction_in_session`. `delta == 0` is intentionally skipped to keep push settlements out of the log. `kind` values come from `typings.economy.TransactionKind`.
- **API surface** (all native async; each function opens `AsyncSession(bind=_engine, expire_on_commit=False)` directly so tests can monkeypatch `_engine` and the swap takes effect immediately):
    - `add_balance(user_id, name, amount)` — **low-level credit primitive**, does not log, does not auto-repay. Used by tests and internal helpers.
    - `credit_with_repayment(user_id, name, amount, kind, note=None)` — message reward / chat reward / casino payout path. **50% of every income event auto-repays outstanding debt** (interest first, then principal). Returns `CreditResult`.
    - `place_bet(user_id, name, requested_bet)` — low-level atomic upfront wager withdrawal retained for tests / internal tools. Slash games do **not** use it anymore. Logs `CASINO_BET`.
    - `settle_game(user_id, name, delta)` — player payout. Clamps to ≥ 0. Logs the *applied* delta (post-clamp) as `CASINO_PAYOUT`.
    - `house_settle(user_id, name, delta)` — dealer-side mirror of player payouts (negated). **Does not clamp at zero** — the bot's balance can and should go negative when the casino is losing. Logs `HOUSE_SETTLE`.
    - `transfer(...)` — atomic conditional debit + credit; returns `TransferResult` or `None`. Logs `TRANSFER_OUT` + `TRANSFER_IN` in the same transaction.
    - `apply_round_settlement(...)` — atomic casino settlement; applies the player's signed net delta only after the round resolves. Positive player deltas go through `credit_with_repayment` (so casino profit auto-repays debt); negative player deltas debit without a zero clamp, so a finished loss can make the player balance negative. Dealer side mirrors through `house_settle`.
    - `borrow(user_id, name, amount, credit_limit_value)` — `principal + interest + amount ≤ credit_limit_value` or rejected. Disburses to balance but **does not** bump `total_earned`. Logs `BORROW`.
    - `repay(user_id, name, amount)` — debits positive user balance, pays interest first then principal. Clamps to `min(amount, balance, debt_total)` and returns `None` when balance is zero or negative. Does **not** bump `total_spent` (repayment isn't gameplay spending; `loan_total_repaid` is the tracking column). Logs `REPAY`.
    - `get_loan_view(user_id)` — read-only `LoanView` snapshot; callers compute pending interest via `accrual_delta(principal=…, last_accrual_at=…, now=…)`.
    - `credit_limit(user)` — pure function; tiered by Discord account age (`User.created_at`, snowflake-derived, free): \<30d→1k, \<180d→10k, \<1y→50k, \<3y→200k, ≥3y→500k. Identical in DMs / guilds / across servers. Inline tier table in the function body — don't extract a module-level constant.
    - `accrual_delta(...)` — pure function; simple 1%/day interest on outstanding principal, `int()`-floored so sub-point fractions accumulate across calls instead of being permanently rounded off (`_accrue_interest_in_session` leaves `loan_last_accrual_at` untouched when the floor is zero).
    - `top_n(limit, exclude_user_ids=())` — `/leaderboard` data source; always called with `(bot.user.id,)` so the house never crowds out real players.
    - `top_debtors(limit, exclude_user_ids=())` — `/debt_leaderboard` data source; computes pending interest at read time and sorts by `principal + interest` without mutating rows.
    - `get_account(user_id)` — used by `/house` to surface gross flows.
- **Bet flow**: slash games validate and clamp the requested bet at start, but do **not** mutate balance until settlement. The round resolves through `cogs/_games/settlement.py:settle_wager(...)`, which passes the signed `player_delta` directly into `apply_round_settlement` and mirrors `-player_delta` into `house_settle`. Regular win applies `+bet`, push applies `0`, loss applies `-bet`, natural Blackjack applies `int(bet * 3 // 2)`. If the bot restarts before settlement, the in-memory round is discarded with no balance change; if a player spends points before a finished loss settles, the debit is still applied and may make the balance negative.
- **Loan flow**: `/borrow` disburses against `credit_limit(user)`; `/repay` pays interest-then-principal from balance. Interest accrues lazily: every loan helper calls `_accrue_interest_in_session` before reading state, so there's no background task. The 50% auto-repay rule applies on every `MESSAGE_REWARD`, `CHAT_REWARD`, and casino payout; `/give` receiver is **not** auto-repaid (gifts shouldn't be confiscated). `/balance` shows `未還本金` / `累積利息` fields (with pending accrual added) when the user has debt, and a `帳號年齡 X 天 · 借款上限 Y` footer either way.

### Games (`cogs/games.py` + `cogs/_games/`)

Slash command `/blackjack`, one game per invocation against an AI dealer. No lobby, no shared table.

- **Pure rules** live in `cogs/_games/blackjack.py`. Side-effect-free; tests inject a seeded `random.Random`, production uses `random.SystemRandom()`.
- **Natural Blackjack**: `is_blackjack` means exactly two cards totaling 21. Player natural settles immediately at 1.5×. Dealer natural also settles immediately unless player also has natural (push). The final embed adds an "提前結束原因" field when this skips the Hit / Stand flow.
- **Shared settlement** lives in `cogs/_games/settlement.py`. `settle_wager(...)` is the single DB-settlement path; `settle_blackjack_round(...)` wraps it for Blackjack. Used by interactive Blackjack (`views.py`) and natural-Blackjack early settles.
- **Response cleanup** lives in `cogs/_games/cleanup.py`. Game response messages are persisted by `(channel_id, message_id)` in `data/game_cleanup.db` as soon as the round message is created, so `GamesCogs.on_ready` can delete stale messages left by a previous bot process. Final casino messages schedule deletion 180 seconds after settlement; zero-balance rejection embeds and game-related economy lookups (`/balance`, `/leaderboard`, `/debt_leaderboard`, `/house`, `/borrow`, `/repay`) schedule deletion after send. `/give` transfer records are intentionally kept. Keep Blackjack timeout settlement separate so abandoned hands still settle before their final embed cleanup starts.
- **Presentation helpers** (`cogs/_games/presentation.py`) centralize outcome labels / colors, all-in wording, bet field text, and the settlement footer. `莊家餘額` is an absolute ledger balance with no leading `+`; only the player round delta shows a sign.
- **`BlackjackView`** drives Hit / Stand. `interaction_check` restricts buttons to `owner_id`; `on_timeout` auto-stands. `_finalize` is guarded by an `asyncio.Lock` + `_settled` flag so Hit / Stand / timeout can't pay out twice.
- **Dealer hint visibility**: the embed hides the dealer's first card and shows the second. `dealer_visible_value(...)` must reflect this so `DealerAI.hint(...)` never sees hidden-card info.
- **Dealer identity is dynamic**: `_dealer_identity()` returns `(bot.user.id, bot.user.display_name)`. Dealer banter goes into `embed.description` *without* a name prefix — the bot is the message sender, so `log_msg.py` records the speaker correctly.
- **House ledger row**: every player settlement mirrors `-player_delta` into the bot's own `UserAccount` row via `house_settle`. Excluded from `/leaderboard`, surfaced separately by `/house`.
- **Dealer banter** is `cogs/_games/dealer.py:DealerAI` — a thin wrapper around an `AsyncOpenAI` client. Three entry points (`taunt_bet`, `settle`, `hint`); each falls back to a hard-coded line on LLM failure so rounds never stall. Prompts live in `cogs/_games/prompts.py` as fixed strings (no `{dealer_name}` placeholder — the dynamic name only flows into the embed). Keep `GameKind` labels in sync when adding games.
- **`GamesCogs.dealer`** is a `cached_property`. Each cog with an LLM client (this one, `ReplyGeneratorCogs`, `AutoUnmuteCogs`) owns its own — three independent clients are intentional, not duplication.

### Threads parsing (`cogs/parse_threads.py`)

Listener that watches `on_message` for Threads URLs (`threads.net` / `threads.com`) and replies with parsed embeds + downloaded videos. Status communicated via reactions (🔗 → 🆗 / ⚠️ / ❌). Adds no extra action reward beyond the global base message reward.

### Video downloader (`cogs/video.py`)

`/download_video` around `utils.downloader.VideoDownloader` (yt-dlp). The `_deliver` helper is shared between the direct path and the "file too big, retry at low quality" fallback — don't duplicate message-build logic in the branches. Pays no points.

**Status text and the file go through different mechanisms on purpose.** Progress text rides on the deferred placeholder via `interaction.edit_original_message(content=...)`; the final file goes out as a fresh `interaction.followup.send(content=..., file=...)`, then the placeholder collapses to `"✅"`. **Discord drops `content` when a multipart file is attached to `edit_original_message`** — an earlier version pushing both at once silently lost the text. Do not collapse `_deliver` back into a single edit.

### Auto-unmute (`cogs/auto_unmute.py` + `cogs/_auto_unmute/prompts.py`)

When a moderator times out the bot itself, this cog detects the `on_member_update` transition to a future-dated `communication_disabled_until`, looks up the moderator via the audit log, clears the timeout, and posts a single sassy AI reply.

- **Reply target**: `_last_active_channel[guild.id]` (updated on every human `on_message`), falling back to `guild.system_channel`. Discord's `member_update` audit entry doesn't carry a channel — this is the only reliable handle.
- **Audit lookup walks 5 entries** because the `member_update` bucket covers nickname / mute / deafen too; pick the entry whose diff carries `communication_disabled_until`.
- `member.edit(timeout=None, …)` fires `on_member_update` again; the early return at the top of the listener prevents an infinite loop.
- `nextcord.Forbidden` (missing `view_audit_log`) is logged and swallowed; the AI gripes at an anonymous moderator.

### Config (`src/discordbot/typings/`)

Each config is a `pydantic_settings.BaseSettings` with `validation_alias=AliasChoices("ENV_NAME")` so env-var names are explicit. `.env` is auto-loaded at import time.

- `DiscordConfig` — `DISCORD_BOT_TOKEN` (required).
- `LLMConfig` — `OPENAI_BASE_URL`, `OPENAI_API_KEY`.
- `ModelSettings` / `RouteDecision` (not env config, same package). `ModelSettings(name, effort)` builds the Responses-API reasoning block and dispatches the right provider's web-search tool. Accepted input modalities are looked up via `utils/model_pricing.get_supported_modalities` (kept out of `typings/` to avoid `utils/` imports).
- `economy.py` — pure frozen `pydantic.BaseModel` types (`LoanView`, `CreditResult`, `BorrowResult`, `RepayResult`) and the `TransactionKind` enum, imported by `cogs/_economy/database.py`. Pure types go here even when they're not env-backed config, as long as they don't pull in `cogs/` or `utils/`.

Keep `Field(description=..., examples=...)` populated when adding configurable values — descriptions are load-bearing here.

### Logging (`src/discordbot/__init__.py`)

`setup_logging()` configures `logfire` with `send_to_logfire=False` (local-only) and tees stdout into `data/logs/<timestamp>.log` via a `_TeeStream` that strips ANSI escape codes. A `LogfireLoggingHandler` is attached to the `nextcord.state` logger. Use `logfire.info / warn / error(..., _exc_info=True)` in new code — avoid stdlib `logging.*`.

### Message logging (`cogs/log_msg.py`)

Every loggable `on_message` is UPSERTed into the `messages` table in `data/messages.db`. The engine is a module-level singleton (`_sql_engine`) — do not move it back onto a per-instance `cached_property` (that was the dominant memory leak). SQLite I/O stays off the event loop via `asyncio.to_thread`; connection PRAGMAs enable WAL + `busy_timeout`.

**Logs human messages AND this bot's own replies — never third-party bots.** The author filter lives in `LogMessageCog._should_log`; `MessageLogger.log` itself is filter-free so a future caller can log any message without re-implementing the gate.

**Streaming bot replies are captured via UPSERT.** Both `on_message` and `on_message_edit` go through the same INSERT keyed by `discord_message_id` with a partial unique index; `ON CONFLICT DO UPDATE` refreshes content/attachments and pins `created_at`. The multi-edit streaming flow in `_handle_streaming` collapses into one row mirroring the final on-Discord state.

### Utilities (`src/discordbot/utils/`)

- `model_pricing.py` — lazy LiteLLM price-table fetch + on-disk cache at `data/model_prices.json`. Exposes `get_token_rates()` (streaming footer) and `get_supported_modalities()` (`_get_attachment_parts` gate). Defaults to `{"text", "image"}` for under-populated upstream entries. Returns `(0.0, 0.0)` for unknown models so the footer shows `$0.00000000` rather than a bogus estimate.
- `images.py` — `get_pil_image` / `get_image_data` / `convert_base64_to_data_uri`. The image-edit path passes `use_b64=False` to get raw `bytes`.
- `downloader.py` — yt-dlp wrapper. Returns a `DownloadResult` context manager that unlinks the file on exit.
- `threads.py` — Threads URL parser / scraper. Normalises `threads.com` → `www.threads.net` and strips query strings.

### Data dir (`data/`)

- `logs/` — per-run tee'd logs.
- `maplestory/` — Artale JSON dataset.
- `downloads/`, `threads/` — ephemeral media scratch (cleaned up by their respective cogs).
- `messages.db` — message log.
- `economy.db` — point balances, loan state, and `point_transaction` audit log.
- `game_cleanup.db` — pending casino response `(channel_id, message_id)` records for restart cleanup.
- `model_prices.json` — cached LiteLLM price table.

## Coding conventions

- **Ruff** is formatter + linter, configured in `pyproject.toml`. Don't blanket-`# noqa`; prefer the narrowest possible `# noqa: <rule>` with a one-line reason.
- **mypy** runs in pre-commit. `Any` is a last resort.
- **Keyword arguments are required for every call**, including single-arg ones (`create_engine(url=...)`, `re.compile(pattern=...)`, `BytesIO(initial_bytes=...)`). Exceptions:
    1. Signature-level positional-only (`Path("a/b")`, exception constructors, `logfire.info("…")`).
    2. Variadic `*args` collectors (`contextlib.suppress(Exception, OSError)`, `AliasChoices("ENV_NAME")`).
    3. One-line stdlib idioms (`len(x)`, `str(x)`, `s.split(",")`).
- **No intermediate one-level aliases** (`usage = responses.usage` → use `responses.usage` directly).
- **`responses` / `response` naming**: LLM SDK return objects are named `responses` (streaming or not); loop variable is `response`.
- **LLM latency matters** in the request path. No extra LLM calls for cosmetic improvements; no executor/`asyncio.gather` scaffolding for ~100 ms CPU work without measuring.
- **Comments**: never narrate what code says or reference PRs / issues.
- **Docs**: `docs/` is generated; `make gen-docs` regenerates it. Don't hand-edit.

## Non-obvious things to remember

- **Do not touch the README badge block.** It may be outdated but is curated.
- **Prompts live only in `cogs/_gen_reply/prompts.py`.** Service logic stays in `gen_reply.py`; don't mass-extract for symmetry.
- DO NOT USE `dataclass`, use `pydantic` instead.
- DO NOT ADD TOO MANY `_*` for global args and module private constant.
- **`AsyncOpenAI` is a `cached_property`** on each cog that needs it. Lazy construction is intentional — moving it into `__init__` would fail at import time in tests where env vars aren't loaded. Three independent clients is intentional.
- **Gemini quirk**: when `reasoning.effort != "none"`, the OpenAI-compat layer prepends `\n\n\n` to streamed text. `_handle_streaming` strips leading newlines on the first delta (`content_started` flag) — don't remove that guard.
- **Gemini thought summary only flows through the Responses API.** Chat Completions silently drops it via LiteLLM. With `reasoning.effort != "none"` and `reasoning.summary` set, the Responses API emits both `reasoning_summary_text.delta` (condensed) and `reasoning_text.delta` (full). Don't swap call sites back to Chat Completions for "simplicity".
- **Responses API role ↔ content-type pairing is strict on current OpenAI models** (Gemini/Claude via LiteLLM are lax, which has masked violations in the past — so the same payload can silently work on the slow model and fail after a swap):
    - `role=user/system/developer` → content parts `input_text` / `input_image` / `input_file`.
    - `role=assistant` → content parts `output_text` / `refusal` only.
    - Build inputs with `EasyInputMessageParam` + `ResponseInputTextParam` / `ResponseInputImageParam` / `ResponseInputFileParam`; `cast("ResponseInputParam", message_list)` only at the SDK boundary.
    - Prefer the string-content shorthand when there are no attachments — the SDK picks the right part type and preserves assistant-role semantic weighting.
    - Any processed Discord message with attachments falls back to `role=user` (since `output_text` can't hold `input_image`/`input_file`); identity is carried via the `{display_name} ({name}) [id: {id}]:` prefix from `_get_cleaned_content`.
    - Separator messages use `role=system`, not `developer`, for Gemini/Claude compatibility via LiteLLM. The real system prompt goes through `instructions`.
- **OpenAI SDK image-edit quirk**: `client.images.edit(image=...)` needs raw `bytes`, not `ImageInputReferenceParam` dicts. `_handle_image_reply` extracts via `get_image_data(use_b64=False)`.
- **Per-token pricing comes from `utils.model_pricing`**, fetched lazily from upstream LiteLLM JSON and cached at `data/model_prices.json`. If a new model shows `$0.00000000`, the upstream JSON hasn't catalogued it yet — wait or delete the cache to force a refresh. Do NOT hardcode rates in `gen_reply.py`.
- **`message.snapshots` (Discord forwards) are intentionally NOT walked** by `_get_cleaned_content` or `_get_attachment_parts`. Forwards are rare; doubling per-message work isn't worth it. Revisit if they become common.
- **`BELIEF` (in `_gen_reply/prompts.py`) is currently disabled** in `_handle_message_reply` — the model treated it as too prescriptive and refused benign requests. The argument still flows through the signature for a future re-enable; don't delete it.
- **Point rewards:** `cli.py` pays the global 5,000-point `MESSAGE_REWARD` for every non-bot `Message`; `gen_reply.py` adds token-based `CHAT_REWARD` only when the bot streams an AI reply. Other cogs should not call `add_balance(...)` or invent per-action rewards.
