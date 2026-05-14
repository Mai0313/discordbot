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
4. **Slow path** (`_handle_message_reply` → `_handle_streaming`): streams `response.output_text.delta` and appends a footer with model name, token usage, cost, and chat-reward balance. The balance comes from `_award_chat_points` (calls `credit_with_repayment`) on `response.completed`; on DB failure the footer degrades gracefully rather than blocking the reply. The first 30 chars create a `reply`; subsequent chunks `edit` it. A `🌐` reaction is added if any `response.output_text.annotation.added` event fires (web search was invoked).
5. **Per-provider tools**: `ModelSettings.tools` dispatches by substring match on `name` to Gemini's `googleSearch`+`urlContext`, Claude's `web_search_*`+`web_fetch_*`, or OpenAI's `web_search`. Extend that property for new providers.
6. **Progress UX**: status is communicated via reactions on the **user's** message (🤔 → 🔀 → 🎨/🎬/📖/❓ → 🆗, plus 🌐 / ❌). The bot never sends an intermediate "thinking…" message — preserve this.
7. **Attachment ingestion** (`_get_attachment_parts`) is gated by `get_supported_modalities(model_name=slow_model.name)`. Attachments whose modality the slow model doesn't accept are dropped before any LLM call. Images go through `_image_to_part` (PIL resize + JPEG re-encode, sent as `input_image` data URI); everything else (`video/*`, `application/pdf`, `text/plain`, …) goes through `_attachment_to_part` as `input_file`. For Discord embeds, `media.discordapp.net` `proxy_url` is preferred over the origin URL since CDN links expire.
8. History / reference / current messages are fetched in parallel via `asyncio.gather`. The tasks list is built in a `for` loop on its own line, then gathered — don't collapse into a comprehension.

### Economy (`cogs/economy.py` + `cogs/_economy/database.py` + `typings/economy.py`)

Persistent point balances backing the global message reward, AI chat reward, casino games, house ledger, loans, daily check-in, VIP, and `/balance` / `/checkin` / `/vip` / `/leaderboard` / `/loss_leaderboard` / `/give` / `/house` / `/borrow` / `/repay`.

- **DB**: SQLite at `data/economy.db` (NOT `messages.db`). The `AsyncEngine` is a module-level singleton (`_engine`). Do not move it onto a `cached_property` — that's the same memory-leak pattern `log_msg.py` already learned. Each helper opens `AsyncSession(bind=_engine, expire_on_commit=False)` directly so tests can monkeypatch `_engine` and the swap takes effect immediately.
- **Schema**:
    - `UserAccount(user_id PK, ...)` — **no `guild_id`**, points are cross-server by design (so `/leaderboard`, loans, VIP, check-in are all cross-server). Loan / VIP / check-in columns live on the same row so message reward / chat reward / casino payout can pay debt and check-in can update streak inside one UPDATE. `name` / `avatar_url` are last-seen Discord metadata refreshed opportunistically on economy writes.
    - `PointTransaction` — append-only audit log; every balance-mutating helper writes one row via `_log_transaction_in_session`. `delta == 0` is intentionally skipped to keep push settlements out of the log. `kind` values come from `typings.economy.TransactionKind`. Indexed on `(occurred_at, kind)` so `top_losers` can scan one day of casino rows cheaply.
    - `JackpotPool(game_id PK, pool_balance, total_contributed, total_claimed, seeded_amount, updated_at)` — per-game shared jackpot, currently only used by Dragon Gate. Seeded by `_ensure_schema` via `ON CONFLICT DO NOTHING`; the seed list lives in `_JACKPOT_SEEDS`. `seeded_amount` is bookkeeping only — the bot's `user_account` row is **not** decremented to fund the seed (it's "on the house"). `apply_jackpot_settlement(player_id, player_delta, game_id)` is the public atomic settle path: player ±delta + pool ∓delta in one session. There's no audit row for the pool itself; the player side still writes a `CASINO_BET` / `CASINO_PAYOUT` row via `_credit_with_repayment_in_session` / `_apply_signed_delta_in_session` (the unified signed-delta helper used for both `CASINO_BET` and `HOUSE_SETTLE` rows).
    - **Legacy migration**: `_ensure_schema()` `ALTER TABLE`-drops obsolete `loan_interest` / `loan_last_accrual_at` columns on startup (older DBs declared them `NOT NULL` without a default), and `ALTER TABLE`-adds `is_vip` / `last_checkin_at` / `checkin_streak` with column-level defaults so existing rows backfill cleanly.
- **Design rules** (read `cogs/_economy/database.py` for full signatures):
    - **50% auto-repay**: `credit_with_repayment` is the income path for message reward / chat reward / casino payout — half of every positive event pays down outstanding principal. `/give` recipients are **not** auto-repaid (gifts shouldn't be confiscated).
    - **No interest, daily reset**: every loan helper calls `_reset_expired_loan_in_session` first; any loan opened before today's Taipei midnight has `loan_principal` (and `loan_opened_at`) cleared as a lazy daily reset before anything else runs.
    - **House ledger can go negative**: `house_settle` mirrors player payouts and does **not** clamp at zero. Player-side `settle_game` (legacy clamp path) clamps at zero, but `apply_round_settlement` (current casino path) lets a finished loss drive the player's balance negative if they spent the bet between deal and settle.
    - **Casino settlement is one atomic step**: slash games validate/clamp the bet up front but don't mutate balance until the round resolves through `cogs/_games/settlement.py:settle_wager`, which applies the signed `player_delta` via `apply_round_settlement` and mirrors `-player_delta` into `house_settle`.
    - **`credit_limit(user, *, is_vip)`** is pure and tiered by Discord account age (snowflake-derived): \<30d→1k, \<180d→10k, \<1y→50k, \<3y→200k, ≥3y→500k. Doubled for VIPs. Inline tier table in the function body — don't extract a module-level constant.
    - **VIP Blackjack bonus** (`apply_vip_blackjack_bonus`) returns `int(delta * 3 // 2)` only when `delta > 0` and `is_vip`. Applied at `settle_wager` time so the rule lives in the game layer.
    - **Check-in concurrency**: `checkin` is SELECT-then-conditional-UPDATE in a retry loop gated on the observed `last_checkin_at`, so two concurrent calls can't double-credit on the same Taipei day. New users go through `INSERT … ON CONFLICT DO NOTHING` and fall back to the UPDATE path on conflict.
    - **Leaderboards exclude the house**: `top_n` is always called with `(bot.user.id,)`. `top_losers` sums `CASINO_BET` + `CASINO_PAYOUT` since today's Taipei midnight and is read-only.
- **`/balance`** shows the `未還本金` field when debt exists, a 👑 VIP badge when applicable, and a `帳號年齡 X 天 · 借款上限 Y · 每天 00:00 Asia/Taipei 重置` footer either way.

### Games (`cogs/games.py` + `cogs/_games/`)

Slash commands `/blackjack` and `/dragon_gate`, each opens a lobby against an AI dealer.

All mini-games support multiplayer. The default flow is: initiator opens a game, the bot shows join buttons for other players, then the initiator starts the game. Add game-specific setup steps in the lobby only when the rules need them. A single player must still be allowed to start so the same commands keep working in DMs.

- **Pure rules** live in `cogs/_games/blackjack.py` and `cogs/_games/dragon_gate.py`. Side-effect-free; tests inject a seeded `random.Random`, production uses `random.SystemRandom()`.
- **Lobby base classes** live in `cogs/_games/lobby.py`. `BaseGameLobbyView(ui.View)` owns the shared join / leave / start scaffold, the participant lock, `on_timeout`, `_send_notice` (with `is_done()` fallback), `on_error` (logfire), `_disable_buttons`, and an optional `max_players` class attr. Subclasses override `_build_lobby_embed(status)` and `_start_game(message)`. `BaseJackpotLobbyView(BaseGameLobbyView)` adds the pre-game ante settlement loop (`_settle_pregame_antes` → `apply_jackpot_settlement` per participant) and an `_jackpot_snapshot` field; subclasses declare `game_id` + `ante` ClassVar and override `_start_game_after_antes(message, final_balances)`. **Do not use `abc.ABC` here** — `nextcord.ui.View.__class__ is type`, so multiple inheritance works, but the project prefers `raise NotImplementedError` for thin base classes. New jackpot game = subclass `BaseJackpotLobbyView` + two hook implementations.
- **`PrepareParticipant` / `RefreshParticipants`** (in `lobby.py`) are the uniform callback protocols both lobby kinds share. Game-specific wager / mode / insufficient-balance embed builders are bound by the caller (`/blackjack` and `/dragon_gate` in `games.py`) via `functools.partial` so the lobby never sees `requested_bet` or game-specific embed copy directly.
- **Natural Blackjack**: `is_blackjack` means exactly two cards totaling 21. Player natural settles immediately at 1.5×. Dealer natural also settles immediately unless player also has natural (push). The final embed adds an "提前結束原因" field when this skips the Hit / Stand flow.
- **Dragon Gate** (射龍門): two pillar cards make an opening, third card resolves the bet. `gate_win` pays 1:1, `pillar_hit` charges 2× into the pot, `outside_lose` charges 1×. Pair pillars require a `higher`/`lower` choice first; `pair_win` pays 1:1, `pair_pillar_hit` (super-pillar-hit) charges **3× into the pot**, `pair_lose` charges 1×.
- **Dragon Gate is a global-jackpot game**: there is no per-round pot. A single row in `jackpot_pool` (`game_id = "dragon_gate"`) is shared across every table; the ante (fixed `ANTE = 5_000`), wins, and losses all flow into / out of that row. The pool is seeded with 100,000 by `_ensure_schema` (one-time, on the house — `seeded_amount` is bookkeeping, the bot's own user_account is **not** decremented). Every settlement (lobby ante, each `place_bet`, leave/timeout refund) is one atomic `apply_jackpot_settlement(...)` call writing player ±delta and pool ∓delta in one SQLite transaction. The `MIN_BET = 10_000` floor and `max_bet = pool` cap are read from the live snapshot held in `DragonGateView._jackpot_snapshot`, refreshed every settlement.
- **Per-player leave + 逆贏不拿**: a `Leave` button (row 2) lets any seated player withdraw without ending the table. On leave / timeout, if `round_state.player_delta(user_id) > 0` (player is ahead since joining), that surplus is refunded back into the jackpot (`apply_jackpot_settlement(player_delta=-delta)`); a non-positive delta is left untouched (losses already flowed into the pool at bet time). The "pool emptied" branch in `_place_bet_locked_by_interaction` deliberately skips this refund — when the pool is naturally cleared, winners keep their winnings. `DragonGateView.interaction_check` opens the Leave button to every non-withdrawn participant while keeping high/low and bet-select restricted to the active turn's player. `DragonGateRound.withdraw(user_id)` advances rotation past withdrawn seats and flips `finished=True` once everyone is out.
- **Shared settlement** lives in `cogs/_games/settlement.py`. `settle_wager(...)` and `settle_blackjack_round(...)` are the Blackjack-only DB paths. Dragon Gate does **not** go through `apply_round_settlement` — it has its own `apply_jackpot_settlement(player_id, player_delta, game_id)` in `cogs/_economy/database.py` that routes the counter-party flow into `jackpot_pool` instead of the dealer ledger.
- **Response cleanup** lives in `cogs/_games/cleanup.py`. Game response messages are persisted by `(channel_id, message_id)` in `data/game_cleanup.db` as soon as the round message is created, so `GamesCogs.on_ready` can delete stale messages left by a previous bot process. Final casino messages schedule deletion 180 seconds after settlement; zero-balance rejection embeds and game-related economy lookups (`/balance`, `/leaderboard`, `/loss_leaderboard`, `/house`, `/borrow`, `/repay`) schedule deletion after send. `/give` transfer records and `/checkin` (ephemeral) / `/vip` responses are intentionally kept. Keep Blackjack timeout settlement separate so abandoned hands still settle before their final embed cleanup starts.
- **Presentation** (`cogs/_games/presentation.py`) centralizes colors, emoji constants, and Markdown helpers (`card_line`, `metadata_line`, `lobby_participant_line`, `player_result_title`, `settlement_metadata`, `build_dealer_talk_embed`). `莊家餘額` is an absolute ledger balance with no leading `+`; only the player round delta shows a sign.
- **Embed layout strategy**: Discord renders `#`/`##`/`###` headings reliably only inside `embed.description` (and not at all inside `inline=True` fields). Any heading-driven content (cards, totals, results) belongs in description; auxiliary text lives in fields or footer. `set_thumbnail(url=owner.avatar_url)` puts the initiator's avatar in the top-right of every game embed (lobby/in-progress/final).
- **Multi-embed messages**: in-progress and final messages send multiple embeds in one `message.edit(embeds=[...], view=...)`. Order is `[dealer talk, main, history?]`. The dealer talk embed (`build_dealer_talk_embed`) uses `set_author(name, icon_url)` so the dealer name+avatar shows in the top-left of that embed. Dragon Gate adds a third history embed built by `build_dragon_gate_history_embed` — a single code block with each turn and a cumulative scoreboard. `DragonGateView.in_progress_embeds()` is the shared helper that builds this list, called by both `DragonGateLobbyView._start_game_after_antes` and `DragonGateView` action callbacks.
- **`BlackjackView`** (in `cogs/_games/blackjack_views.py`) drives Hit / Stand. `interaction_check` restricts buttons to the active player; `on_timeout` auto-stands every unresolved player. `finalize` is guarded by an `asyncio.Lock` + `_settled` flag so Hit / Stand / timeout can't pay out twice. The view also accepts a `dealer_line` parameter so the lobby's `taunt_bet` line flows into the first dealer talk embed (don't revert to the hard-coded default). `BlackjackLobbyView` is a thin subclass of `BaseGameLobbyView` (`max_players = MAX_BLACKJACK_PLAYERS`); it only adds `requested_bet` / `dealer_id` fields and implements `_build_lobby_embed` + `_start_game`.
- **`DragonGateView`** drives high/low button + bet StringSelectMenu + the per-player Leave button. The Row 0 buttons (`同點猜大`/`同點猜小`) only enable when the current pillars are a pair and direction is unchosen; selected direction is marked with ` ✓` suffix. The Row 1 StringSelectMenu (`custom_id="dg:bet"`) has three options: `底注 X` (min), `全池 X` (max), `自訂` (opens `DragonGateBetModal`). Row 2 holds `離桌` (`custom_id="dg:leave"`) which is gated by `is_active(user_id)` and runs the 逆贏不拿 refund + rotation advance via `round_state.withdraw(user_id)`. The `_handle_bet_choice(choice, interaction)` helper routes the select value; tests call it directly instead of going through Discord's component plumbing. `sync_controls` updates button labels/disabled states and the select's options/placeholder/disabled — including a fresh min/max read from `_jackpot_snapshot` — in lockstep after every settlement.
- **Dealer hint visibility**: the embed hides the dealer's first card and shows the second. `dealer_visible_value(...)` must reflect this so `DealerAI.hint(...)` never sees hidden-card info.
- **Dealer identity is dynamic**: `_dealer_identity()` returns `(bot.user.id, bot.user.display_name, bot.user.display_avatar.url)`. Dealer banter goes into the dedicated dealer talk embed (`build_dealer_talk_embed`) with bot avatar as the embed author icon — `log_msg.py` records the speaker correctly because the message is still sent by the bot account.
- **House ledger row**: every **Blackjack** player settlement mirrors `-player_delta` into the bot's own `UserAccount` row via `house_settle`. Excluded from `/leaderboard`, surfaced separately by `/house`. Dragon Gate **does not** touch the house ledger — its counter-party is `jackpot_pool`, so `/house` numbers reflect Blackjack P&L only.
- **Dealer banter** is `cogs/_games/dealer.py:DealerAI` — a thin wrapper around an `AsyncOpenAI` client. Four entry points (`taunt_bet`, `settle`, `hint`, `table_settle`); each falls back to a hard-coded line on LLM failure so rounds never stall. Prompts live in `cogs/_games/prompts.py` as fixed strings (no `{dealer_name}` placeholder — the dynamic name only flows into the embed). Keep `GameKind` labels in sync when adding games.
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

Each config is a `pydantic_settings.BaseSettings` with `validation_alias=AliasChoices("ENV_NAME")` so env-var names are explicit. `.env` is auto-loaded at import time. `ModelSettings(name, effort)` builds the Responses-API reasoning block and dispatches the right provider's web-search tool; accepted input modalities are looked up via `utils/model_pricing.get_supported_modalities` (kept out of `typings/` to avoid `utils/` imports).

`typings/economy.py` holds pure frozen `pydantic.BaseModel` types, the `TransactionKind` enum, and tunable constants (`BASE_MESSAGE_REWARD_AMOUNT`, `BASE_CHECKIN_REWARD_AMOUNT`, `CHECKIN_STREAK_CYCLE`, `VIP_PURCHASE_COST`). Pure types go here even when they're not env-backed config, as long as they don't pull in `cogs/` or `utils/`. Single-caller numeric coefficients (e.g. the day-1→day-7 streak step) are inlined into their function body, not exported.

Keep `Field(description=..., examples=...)` populated when adding configurable values — descriptions are load-bearing here.

### Logging (`src/discordbot/__init__.py`)

`setup_logging()` configures `logfire` with `send_to_logfire=False` (local-only) and tees stdout into `data/logs/<timestamp>.log` via a `_TeeStream` that strips ANSI escape codes. A `LogfireLoggingHandler` is attached to the `nextcord.state` logger. Use `logfire.info / warn / error(..., _exc_info=True)` in new code — avoid stdlib `logging.*`.

### Message logging (`cogs/log_msg.py`)

Every loggable `on_message` is UPSERTed into the `messages` table in `data/messages.db`. The engine is a module-level singleton (`_sql_engine`) — do not move it back onto a per-instance `cached_property` (that was the dominant memory leak). SQLite I/O stays off the event loop via `asyncio.to_thread`; connection PRAGMAs enable WAL + `busy_timeout`.

**Logs human messages AND this bot's own replies — never third-party bots.** The author filter lives in `LogMessageCog._should_log`; `MessageLogger.log` itself is filter-free so a future caller can log any message without re-implementing the gate.

**Streaming bot replies are captured via UPSERT.** Both `on_message` and `on_message_edit` go through the same INSERT keyed by `discord_message_id` with a partial unique index; `ON CONFLICT DO UPDATE` refreshes content/attachments and pins `created_at`. The multi-edit streaming flow in `_handle_streaming` collapses into one row mirroring the final on-Discord state.

### Utilities (`src/discordbot/utils/`)

- `model_pricing.py` — lazy LiteLLM price-table fetch + on-disk cache at `data/model_prices.json`. Exposes `get_token_rates()` and `get_supported_modalities()`. Defaults to `{"text", "image"}` for under-populated upstream entries. Returns `(0.0, 0.0)` for unknown models so the footer shows `$0.00000000` rather than a bogus estimate.
- `images.py` — the image-edit path passes `use_b64=False` to get raw `bytes` (not base64).
- `downloader.py` — yt-dlp wrapper. `DownloadResult` is a context manager that unlinks the file on exit.

### Data dir (`data/`)

- `messages.db` — message log.
- `economy.db` — point balances, daily-resetting loan principal, VIP flag, check-in streak, `point_transaction` audit log, `jackpot_pool` (per-game shared jackpots; only Dragon Gate currently registered).
- `game_cleanup.db` — pending casino response `(channel_id, message_id)` records for restart cleanup.
- `model_prices.json` — cached LiteLLM price table.
- `downloads/`, `threads/` are ephemeral scratch cleaned up by their respective cogs.

## Coding conventions

- **Ruff** is formatter + linter, configured in `pyproject.toml`. Don't blanket-`# noqa`; prefer the narrowest possible `# noqa: <rule>` with a one-line reason.
- **mypy** runs in pre-commit. `Any` is a last resort.
- **Keyword arguments are required for every call**, including single-arg ones (`create_engine(url=...)`, `re.compile(pattern=...)`, `BytesIO(initial_bytes=...)`). This is a call-site style rule: write `f(a=1, b=2)` when calling functions, but do not add a bare `*` in new function signatures solely to force keyword-only arguments. Prefer `def f(a: int, b: int) -> None:` over `def f(*, a: int, b: int) -> None:` unless an external API or correctness issue explicitly requires keyword-only parameters. Exceptions:
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
