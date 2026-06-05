# CLAUDE.md

Project-specific guidance for Claude Code when working in this repository.

## Commands

All tooling runs through `uv`.

```bash
uv run discordbot                # run the bot
uv run pytest                    # tests, coverage gate: 80%
uv run pre-commit run -a         # canonical pre-push check (= make fmt)
make gen-docs                    # regenerate docs/ from sources
```

## Runtime Shape

- `src/discordbot/cli.py` defines `DiscordBot(commands.Bot)`. Intents are `Intents.all()` minus `members` and `presences`.
- Cog loading is synchronous inside `DiscordBot.__init__` before gateway connection. Every cog module must expose sync `def setup(bot): ...` and add cogs with `override=True`; `async def setup` breaks the first command sync because nextcord schedules it without awaiting.
- Helper packages live in sibling `_<cog>/` directories so they are not auto-loaded as cogs.
- Bot presence lines rotate from the `bot_status` table in `data/database/global_state.db`, managed offline via `scripts/manage_bot_status.py`.
- Add common command errors in `DiscordBot.on_command_error` instead of catching them in each cog.

## Cog Rules

- One `commands.Cog` subclass plus a sync `setup` per cog module. Cogs do not import peer cogs; use the bot instance, shared typings, or cog-private helper packages.
- User-facing slash commands need localized names and descriptions for English, Traditional Chinese, and Japanese.
- Any user-visible command or behavior change must update `src/discordbot/cogs/help.py` in the same change and keep `tests/test_help.py` passing.
- When multiple embeds in one message need aligned widths, use `utils.discord_embeds.embed_spacer_payload(...)` with `target=` so edits retain the already-uploaded spacer; re-uploading it on every edit trips Discord error 400009 on rapidly edited messages.
- When content cannot fit Discord embed limits, paginate or render a PNG referenced via `attachment://...`; never truncate silently.

## AI Pipeline

- Runtime LLM backend is LiteLLM Proxy behind `OPENAI_BASE_URL`; all runtime conversations use `AsyncOpenAI` and the OpenAI Responses API. Do not switch back to Chat Completions, and do not import provider-native SDKs (`google-genai`, `anthropic`) into runtime request paths; `scripts/prompt_dev.py` is the local-experimentation exception.
- Runtime model strings live in `RuntimeModelCatalog` in `src/discordbot/typings/models.py`; keep `slow_model`'s peak-hour dispatch. Provider selection happens through LiteLLM via model strings and `extra_body`.
- Streaming SDK objects are named `responses`; loop items are named `response`.
- AI progress is communicated with reactions on the user's message. Preserve the no-intermediate-message UX.
- Attachment ingestion is gated by `get_supported_modalities` for the slow model; unsupported attachments are dropped before any LLM call. Images are resized and JPEG re-encoded into `input_image` data URIs; other supported attachments are sent as `input_file`.
- For Discord embeds, prefer `media.discordapp.net` `proxy_url` over origin URLs because CDN links expire.
- History, referenced messages, and current messages are fetched with `asyncio.gather`; build the task list in a `for` loop on its own lines.

### Responses API Gotchas

- Strict role and content part pairing: `user` / `system` / `developer` use `input_*` parts; `assistant` uses `output_text` or `refusal`. Messages with attachments therefore fall back to `role=user`. Use `EasyInputMessageParam` plus concrete part types and cast only at the SDK boundary; prefer string-content shorthand when there are no attachments.
- Separator messages use `role=system`, not `developer`, for Gemini and Claude compatibility through LiteLLM.
- Gemini may prepend leading newlines to streamed reasoning output; `_handle_streaming` strips them on the first content delta. Gemini thought summaries only flow through the Responses API.
- `client.images.edit(image=...)` needs raw `bytes`, not image-reference dicts.
- Per-token pricing comes from `utils.model_pricing` and its cached LiteLLM JSON; do not hardcode rates in `gen_reply.py`.
- `message.snapshots` are intentionally not walked by cleaned-content or attachment ingestion.

## Economy

- SQLite files are separate and all live under `data/database/`: `messages.db` (message logs), `economy.db` (per-user 虛擬歡樂豆), `global_state.db` (jackpot pools, casino ledger, presence rotation), `game_cleanup.db` (cleanup targets).
- `cogs/_economy/database.py` owns the module-level async engines `_engine` and `_global_state_engine`; do not move them to `cached_property` because tests monkeypatch them.
- `UserAccount` has no `guild_id`; identity, VIP, check-ins, and flags are cross-server by design. Balances live in `user_wallet`, loans in `loan_proposal` / `loan_contract`, daily casino counters in `casino_account`. Economy money columns are decimal strings parsed to Python `int` so they avoid SQLite's 64-bit integer ceiling.
- Money inputs are string `SlashOption`s parsed by the bot (`_parse_positive_amount`, `_parse_collect_amount`), not integer options; malformed text returns `build_invalid_amount_embed` ephemerally before any mutation. Prefer this pattern for any new numeric input that can exceed Discord limits.
- `UserAccount.avatar_url` is a last-seen cache written via `utils.avatars.guild_avatar_url(...)` with guild context; do not backfill existing URLs.
- There is no transaction table. Every applied positive delta must increase `UserWallet.total_earned`, every negative delta `total_spent`; keep `total_earned - total_spent == balance`.
- `credit_with_repayment` is the income path for message reward, chat reward, and casino payout. Passive income and `/give` recipients do not auto-repay debt.
- Loans: personal credit debits the lender on acceptance; central-bank loans mint on approval and burn on repayment / collection. Pending proposals expire after 180 seconds. Simple interest accrues lazily; acceptance prepays `MIN_INTEREST_DAYS` of interest, so do not zero `interest_due` at contract creation.
- Central banker access (`UserAccount.is_central_banker`, via `scripts/manage_central_banker.py`) is separate from Discord admin. Keep `ECONOMY_ALLOW_CENTRAL_BANK_SELF_APPROVAL` unset or `false` in production.
- Admin adjustments use `adjust_balance(..., allow_negative=...)`, not casino settlement helpers.
- `/leaderboard` keeps DB-side integer-aware ordering over `StoredInteger` text with `LIMIT` before rows reach Python; write paths invalidate caches via `invalidate_economy_leaderboard_cache()`. `hide_from_leaderboard` accounts are omitted.
- Daily casino counters on `casino_account` update only from player-side Blackjack and Dragon Gate settlement deltas; `/loss_leaderboard` reads gross `daily_loss`, so wins do not offset it. The bot is just another player here.
- Cumulative casino-system P&L lives in the `casino_ledger` row in `data/database/global_state.db`. `/casino` reads the ledger; `/pocat` reads the bot's wallet; do not reuse the bot's `user_wallet` as the house ledger. On settlement the player wallet commits before the casino ledger so user balance wins if only one commit succeeds.
- Visibility boundary: shared social / market / settlement events are public embeds with scheduled cleanup (e.g. `/give`, leaderboards, `/casino`, loan flows); personal state and validation / permission failures are ephemeral (e.g. `/balance`, `/checkin`, insufficient balance).
- Rewards: `cli.py` grants the cooldown-gated per-message reward; `gen_reply.py` adds the token-based chat reward after streamed replies. Other cogs must not invent action rewards.
- Anti-inflation constants live in `typings/economy.py`: `transfer()` burns `TRANSFER_TAX_BPS` of every `/give` as a permanent sink, every casino wager caps at `MAX_SINGLE_BET` at the shared chokepoints, and the VIP casino multiplier is 1.2x.
- `scripts/reset_economy.py` is the offline reset (bot stopped, back up the `.db` files, always `--dry-run` against a copy first). It writes through `_economy/database.py` helpers so StoredInteger text stays canonical.

## Stocks

- Simulated stock state lives in `data/database/stock.db`; wallet cash stays in economy `user_wallet`.
- `/stock` is one public market message edited in place for every view; only the opener operates its controls; it deletes itself after 180 idle seconds.
- `stock_profile` is the source of truth for virtual companies; maintain it offline via `uv run python scripts/manage_stock_company.py ...`. Do not seed companies from runtime code and do not add legacy schema migrations.
- `float_shares` caps aggregate long exposure and short borrow capacity; opening new long also clamps each user to 49% of float. Risk-reducing flow (sell long, cover short) is always allowed.
- `stock_news` refreshes at most once per `news_cadence_hours` per symbol. `StockNewsAI` may use the LLM, but database helpers must have deterministic fallback news and never block settlement on LLM failure. Prompts live in `_stock/prompts.py`; keep news fictional, absurd, harmless.
- News sentiment is an impulse applied exactly once at its firing tick boundary, never a decayed drift over following ticks; the decayed value only feeds the AI news prompt as ambient context.
- Market pressure uses decayed order flow and per-stock liquidity; do not reintroduce a multi-day aggregate pressure term.
- Global anti-inflation guardrails are constants in `cogs/_stock/market.py` (`MARKET_VOLATILITY_SCALE_BPS`, `GLOBAL_MAX_TICK_CHANGE_BPS`, Taiwan-style `apply_daily_price_limit`); re-measure with `uv run python scripts/simulate_stock_market.py` after touching them.
- Execution price applies order-size slippage from `liquidity_shares`, capped by `max_tick_change_bps`; store and display per-leg `price_cents`, never settle large orders at the quote price.
- Stock settlement must go through `settle_stock_operation(...)`; views must not split price / wallet / position reads and writes or import stock ORM models. Compound operations write one `stock_operation` plus ordered `stock_trade_leg` rows; do not net wallet legs.
- Money and share columns are decimal strings parsed to `int`; wallet deltas stay integer via `cash_ceil(...)` / `cash_floor(...)`. Tables persisting `user_id` also persist `user_name`; UI displays stored names, not Discord IDs.
- Portfolio views have a short process cache invalidated by stock writes; market board and 7D chart PNG caches are digest-keyed, so include every pixel-affecting field in the render key.
- Lazy ticks advance every 5 minutes on interaction; backlogs compress to `MAX_TICKS_PER_INTERACTION` ticks and still roll Asia/Taipei day boundaries.
- Maintenance: `scripts/manage_stock_reconciliation.py list` for non-final operations, `scripts/manage_stock_company.py audit` for float / exposure / capacity.
- For Discord UI beyond embed and Markdown limits, render a PNG with Pillow and reference it via `attachment://...`; shared CJK font loading and text-anchoring helpers live in `utils/pil_text.py`.

## Games

- Pure rules live in `cogs/_games/blackjack.py` and `cogs/_games/dragon_gate.py`; production uses `random.SystemRandom`, tests inject seeded `random.Random`.
- Lobby scaffolding lives in `cogs/_games/lobby.py`. Keep `raise NotImplementedError`; do not convert base views to `abc.ABC`.
- Casino settlement is one atomic step after the round resolves; validate or clamp bets up front, then settle through the game settlement helpers.
- Blackjack supports Hit, Stand, Double Down, Split, Surrender, Insurance, and peek. `bet=0` means all in; `bet` is a string option. Split matches Blackjack value (10/J/Q/K interchangeable); no Double after Split; split-hand 21 is not natural. `MAX_BLACKJACK_PLAYERS = 6`.
- Settlement uses `BlackjackHandState` plus `BlackjackHandSettlement` / `BlackjackPlayerSettlement`; do not reintroduce the legacy single-hand path.
- 過五關: a hand reaching five+ cards without busting auto-stands; non-21 pays a normal win regardless of dealer total. Five-card 21 adds an extra 1x system-funded bonus that counts as player-side payout but does not move `/casino`. The VIP bonus is `max(0.2x dealer-paid win, 0.2x five-card bonus)`, not the sum.
- The dealer is the casino system, fully deterministic under H17, never an LLM.
- The bot account joins every table as a regular player (auto-added when its wallet > 0) and settles like any human. Its decisions are deterministic, never LLM-chosen: fractional-Kelly bet sizing (`kelly_bet`, edge adjusted by the channel shoe's Hi-Lo true count), the EV engine's hole-aware `recommended_action` (`choose_bot_action`), and count-based insurance (take iff P(ten) > 1/3). Re-measure the Kelly / edge constants with `uv run python scripts/simulate_bot_blackjack.py`.
- `BotPlayerAI` only narrates `reason` text off the critical path: the deterministic template reason shows immediately and the LLM upgrade is a revision-guarded background refresh. The model-facing context never includes the dealer hole card or anything derived from it; the per-action EVs shown to the model come from `blackjack_ev.py`'s marginal pass and are hole-independent. Prompts in `cogs/_games/prompts.py` are narration-only; do not turn them back into decision-making or strategy lookup tables.
- The shoe (default 4 decks) is persistent per channel via `cogs/_games/shoe.py::BlackjackShoeStore` (in-memory, rebuilt past `RESHUFFLE_THRESHOLD_CARDS`) so card counting has signal. Tests needing deterministic draws clear `round_state.shoe` before monkeypatching `draw_card`.
- Action buttons are presence-based: invalid controls are removed from the view, not left visible and disabled.
- Hit hints and settlement banter are background LLM refreshes; player interactions and final result publication must not wait on LLM latency.
- Peek runs as a two-stage animation (`_animate_peek_reveal_bj_locked` / `_animate_peek_no_bj_locked`); do not collapse it into a single edit.
- `BlackjackView._finalize_locked` disables buttons and calls `self.stop()` before settlement; the final edit removes controls with `view=None`.
- Player losses clamp at balance 0; `apply_round_settlement` mirrors only the actual collected debit into `casino_ledger`.
- Dragon Gate is backed by the shared `jackpot_pool` row `game_id="dragon_gate"` in `data/database/global_state.db`, not the casino ledger. Ante, losses, wins, and refunds settle through jackpot helpers; losses clamp at 0; on leave or timeout, positive running delta refunds into the pool unless the whole-pool win already cleared it.
- Interactive views use 180-second idle timeouts; terminal public messages schedule deletion through `utils.message_cleanup`. No cog-local delete loops.
- Game messages send one embed per seat: `[talk, dealer_seat, *player_seat_embeds]` via `build_in_progress_embeds` / `build_final_embeds`. Markdown headings render reliably only inside `embed.description`. Player seats set the avatar thumbnail; the dealer seat omits it. Dealer seat color follows the casino-vs-table outcome: any player win red, all-loss green, all-push yellow.
- The "dealer" is a label, not a Discord identity. `SystemNarrator` produces neutral broadcast lines sent by the bot account so `log_msg.py` records the speaker correctly.
- `GamesCogs.narrator` and `GamesCogs.bot_player_ai` are `cached_property`s; each cog with an LLM client owns its own client intentionally.

## Fishing

- `/games fishing` is a single-player, LLM-free money sink under `cogs/_fishing/`; state lives in `data/database/fishing.db`, wallet cash stays in economy.
- One public message edited in place, opener-only, self-deletes after 180s idle. Mirror `_stock/views.py`: `FishingPublicView` plus the central `edit_fishing_message(...)`.
- `fish_grade_config`, `fish_species`, and `fishing_gear` are the tunable source of truth; seed offline with `uv run python scripts/manage_fishing.py seed-defaults`. `_fishing/defaults.py::build_default_catalog` is the one catalog definition.
- `_fishing/catch.py` is pure and RNG-injected. Luck is the additive rod+bait `rarity_shift_bps`, clamped, never moving the most common grade.
- Net-deflationary by design: every cast's EV is below bait + amortized rod cost, and catches cap at `FISHING_MAX_SINGLE_CATCH`. Re-measure with `uv run python scripts/simulate_fishing.py`; every rod+bait combo must stay a sink. Purchases burn via `apply_ordered_wallet_deltas`; payouts credit via `credit_with_repayment`.
- Cross-DB ordering: `purchase_gear` debits the wallet first, then grants gear (refund on grant failure); `settle_cast` writes fishing.db first, then credits economy.db, returning `CastStatus.PAYOUT_DEFERRED` on payout failure (never rolled back).
- Settlement only via `settle_cast` / `purchase_gear`; views never import ORM models. `catch_log` denormalizes species fields so catalog retuning never rewrites history.

## Memory

- Per-user long-term memory is markdown under `data/memories/<user_id>/` (`main.md`, `raw.md`, `main.bak.md` as recovery point), file-based on purpose and keyed by user id (cross-server, shared with DMs). Do not move it into SQLite or add legacy-layout fallbacks.
- Two-phase pipeline in `cogs/_memory/`: fire-and-forget `schedule_memory_update` runs phase-1 extraction after each streamed reply (per-user in-flight de-dupe replays only the newest skipped turn); phase-2 consolidation rewrites the main file once `RAW_CONSOLIDATION_THRESHOLD` accumulates. Both phases use `RuntimeModelCatalog.memories_model`. Memory must never add latency to the reply path; LLM failures silently keep the previous state.
- Anti-injection invariants: the transcript renders column-0 `[message <n> | <role>]` markers with indented content, `sanitize_identity` neutralizes `[id:`-lookalikes in names, prompts treat conversation content as data rather than instructions, and `redact_secrets` (shape-specific patterns, no bare-hex rule) runs on the transcript and every model output. Keep the transcript structure and the matching rules in `_memory/prompts.py` together.
- The read path injects only the trigger user's main file as a low-trust `role=system` separator at the END of the reply `message_list`, never into top-level `instructions`. Pass the memory-free `message_list` to `schedule_memory_update` so extraction never re-ingests stored memory.
- The main file starts with `v1` and stays under `MAIN_FILE_MAX_CHARS`. `MEMORY_ENABLED` is the kill switch for both paths; `/memory clear` is additionally gated by `MEMORY_CLEAR_ENABLED` (default false); keep the real clear code behind the flag.
- All writes go through `_memory/store.py` under `store.user_lock(user_id)`; main-file writes are atomic and back up to `main.bak.md` first; in-flight updates check `cleared_since` so a clear is not resurrected. Tests isolate state with the `memory_isolated_dir` fixture.
- Memory files carry a single-line author identity for human inspection only; read helpers strip it so prompt injection, consolidation input, and `/memory show` never see it.

## Other Cogs

- `parse_threads.py` watches `threads.net` / `threads.com` URLs and uses reactions for status; no extra reward.
- `video.py` keeps progress text on the deferred original message, then edits that same message with the final file and source URL.
- Temporary media downloads (`video.py`, `parse_threads.py`, `utils/downloader.py`, `utils/threads.py`) write to the project-root `tmp/` scratch folder (gitignored, removed by `make clean`) and delete the files after delivery. Do not put scratch downloads under `data/`.
- `auto_unmute.py` clears timeouts applied to the bot, finds the moderator from recent audit log entries, and replies in the last active human channel or the guild system channel.
- `log_msg.py` logs human messages and this bot's own replies, never third-party bots; streaming replies converge via UPSERT on `discord_message_id`. It owns the module-level `_sql_engine`; do not move it to a cached property.

## Config And Types

- Environment-backed config classes use `pydantic_settings.BaseSettings` with explicit `validation_alias=AliasChoices("ENV_NAME")`. `.env` is loaded at import time.
- Pure shared result types, enums, and constants live under `src/discordbot/typings/` when they do not depend on cogs or utils.
- Use `pydantic.BaseModel` for structured data and type-managed payloads. Do not introduce `dataclass`, and do not use loose `object` / `Any` annotations.
- Keep `Field(description=..., examples=...)` populated for configurable values.

## Coding Conventions

- Ruff is formatter and linter; mypy runs in pre-commit. Use narrow `# noqa: <rule>` comments with a reason. Type hints use the real domain type.
- Keyword arguments are required for normal calls, including single-argument calls; do not add bare `*` to force keyword-only signatures. Allowed positional idioms: `len(x)`, `str(x)`, `Path("x")`, exception constructors, variadic collectors, `logfire.info("message")`.
- Avoid intermediate one-level aliases such as `usage = responses.usage`.
- Import functions by name and call them unqualified (`from pkg.mod import do_thing`); a module-namespace import is a last resort with a reason.
- LLM latency matters; do not add extra LLM calls for cosmetic improvements.
- Comments explain non-obvious behavior only.
- `docs/` is generated by `make gen-docs`; do not hand-edit it. Do not touch the README badge block unless explicitly asked.

## Documentation Split

- `README.md` is the concise canonical user-facing README; `README.zh-CN.md` / `README.zh-TW.md` mirror its structure; `CONTRIBUTING.md` is developer-facing English; `CLAUDE.md` stays dense and AI-agent-facing.
