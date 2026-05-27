# CLAUDE.md

Project-specific guidance for Claude Code when working in this repository.

## Commands

All tooling runs through `uv`.

```bash
uv run discordbot                # run the bot
uv run pytest                    # tests, coverage gate: 80%
uv run pre-commit run -a         # canonical pre-push check
make fmt                         # same as pre-commit run -a
make gen-docs                    # regenerate docs/ from sources
```

## Runtime Shape

- `src/discordbot/cli.py` defines `DiscordBot(commands.Bot)`.
- Intents start from `Intents.all()`, then disable `members` and `presences`.
- Cog loading is synchronous inside `DiscordBot.__init__` before gateway connection. First `on_ready` syncs global application commands and starts the status task.
- Every cog module must expose sync `def setup(bot): ...` and add cogs with `override=True`. Do not use `async def setup`; nextcord schedules it without awaiting, so the first command sync can see no commands.
- Helper packages live in sibling `_<cog>/` directories so they are not auto-loaded as cogs.
- Add common command errors in `DiscordBot.on_command_error` instead of catching them in each cog.

## Cog Rules

- Cog modules should define one `commands.Cog` subclass plus a sync `setup`.
- Cogs should not import peer cogs directly. Use the bot instance, shared typings, or cog-private helper packages.
- Slash commands need localized names and descriptions for English, Traditional Chinese, and Japanese where the command is user-facing.
- Any user-visible command or behavior change must update `src/discordbot/cogs/help.py` in the same change and keep `tests/test_help.py` passing.

## AI Pipeline

- Runtime LLM backend is LiteLLM Proxy behind `OPENAI_BASE_URL`; all runtime conversations use `AsyncOpenAI` and the OpenAI Responses API. Do not switch chat, routing, or captioning back to Chat Completions.
- Provider selection happens through LiteLLM via model strings, `ModelSettings.tools`, and `extra_body`.
- Do not import provider-native SDKs such as `google-genai` or `anthropic` into runtime request paths. `scripts/prompt_dev.py` is the exception for local experimentation.
- Runtime model strings live in `RuntimeModelCatalog` in `src/discordbot/typings/models.py`; update that catalog and keep `slow_model`'s peak-hour dispatch.
- Streaming SDK objects are named `responses`; loop items are named `response`.
- AI progress is communicated with reactions on the user's message. Preserve the no-intermediate-message UX.
- Attachment ingestion is gated by `get_supported_modalities` for the slow model. Unsupported attachments should be dropped before any LLM call.
- Images are resized and JPEG re-encoded into `input_image` data URIs. Other supported attachments are sent as `input_file`.
- For Discord embeds, prefer `media.discordapp.net` `proxy_url` over origin URLs because CDN links expire.
- History, referenced messages, and current messages are fetched with `asyncio.gather`. Keep the task list built in a `for` loop on its own lines.

### Responses API Gotchas

- Current OpenAI models are strict about role and content part pairing: `user` / `system` / `developer` use `input_text`, `input_image`, and `input_file`; `assistant` uses `output_text` or `refusal`.
- Use `EasyInputMessageParam` plus the concrete Responses content part types, then cast only at the SDK boundary.
- Prefer string-content shorthand when there are no attachments.
- Processed Discord messages with attachments fall back to `role=user` because assistant content cannot contain `input_image` or `input_file`.
- Separator messages use `role=system`, not `developer`, for Gemini and Claude compatibility through LiteLLM.
- Gemini may prepend leading newlines to streamed reasoning output when `reasoning.effort != "none"`. `_handle_streaming` strips them on the first content delta; keep that guard.
- Gemini thought summaries only flow through the Responses API.
- `client.images.edit(image=...)` needs raw `bytes`, not image-reference dicts.
- Per-token pricing comes from `utils.model_pricing` and its cached LiteLLM JSON. Do not hardcode rates in `gen_reply.py`.
- `message.snapshots` are intentionally not walked by cleaned-content or attachment ingestion.

## Economy

- SQLite files are separate: `data/messages.db` for message logs, `data/economy.db` for per-user 虛擬歡樂豆, `data/global_state.db` for bot-wide shared state such as jackpot pools, and `data/game_cleanup.db` for public message cleanup targets.
- `cogs/_economy/database.py` owns the module-level async engines `_engine` and `_global_state_engine`. Do not move them to `cached_property`; tests monkeypatch those engines and expect helpers to bind sessions directly from the current object.
- `UserAccount` has no `guild_id`. Identity, VIP, check-ins, admin flags, central banker flags, and leaderboard visibility are cross-server by design. Spendable balances and gross totals live in `user_wallet`, which also denormalizes `name` for direct DB inspection; long-term loans live in `loan_proposal` and `loan_contract`; daily casino counters live in `casino_account`. Economy money columns are stored as decimal strings and parsed to Python `int` for arithmetic so they do not inherit SQLite's 64-bit integer ceiling.
- `UserAccount.avatar_url` is a last-seen cache. Discord-facing write paths use `utils.avatars.guild_avatar_url(...)` with guild context so guild avatars are stored when available, falling back to global `display_avatar`. Do not backfill existing avatar URLs; they refresh naturally on later writes.
- There is no per-mutation `point_transaction` table. Every applied positive balance delta must increase `UserWallet.total_earned`; every applied negative balance delta must increase `UserWallet.total_spent`. Keep `total_earned - total_spent == balance`.
- `credit_with_repayment` is the income path for message reward, chat reward, and casino payout. Long-term loans are repaid explicitly; passive income and `/give` recipients, including bot recipients, do not auto-repay debt.
- Loan helpers store proposal state in `loan_proposal` and debt state in `loan_contract`. Personal credit requests are borrower-initiated and debit the lender on acceptance; central-bank loans mint borrower balance on approval and burn borrower balance on repayment / collection. Pending proposals expire after 180 seconds and are marked `rejected`. Simple interest accrues lazily, daily prorated from the monthly bps rate. Acceptance prepays `MIN_INTEREST_DAYS` of interest into `interest_due` and pushes `last_interest_accrued_at` past that window, so borrow-then-immediately-repay still owes the minimum lock-in interest; do not zero `interest_due` at contract creation.
- Central banker access is separate from Discord admin access. Use `scripts/manage_central_banker.py` for the `UserAccount.is_central_banker` flag. Central-bank borrow requests are public messages with approval / rejection / cancel buttons. `ECONOMY_ALLOW_CENTRAL_BANK_SELF_APPROVAL=true` enables borrower self-approval for local testing; keep it unset or `false` in production.
- Admin adjustments use `adjust_balance(..., allow_negative=...)`, not casino settlement helpers, and may target the bot account.
- `UserAccount.hide_from_leaderboard` defaults to `False`. When set, `/leaderboard` and `/loss_leaderboard` omit that account; maintenance callers can still opt into hidden rows.
- `/leaderboard` must keep DB-side integer-aware ordering over `StoredInteger` decimal text and apply `LIMIT` before rows reach Python. Balance and casino write paths invalidate process-local leaderboard row and PNG caches after successful commits through `invalidate_economy_leaderboard_cache()`.
- Daily casino counters live on `casino_account` as `day_started_at`, `daily_loss`, `daily_win`, and `daily_net`. Player-side Blackjack and Dragon Gate settlements update them from the actual applied delta; the casino ledger row, push rounds, manual adjustments, transfers, loans, rewards, check-ins, and VIP purchases do not. Blackjack five-card 21 system bonuses count as player-side casino payout and daily win/net, but do not move `/casino`. `/loss_leaderboard` reads gross `daily_loss`, so wins do not offset the displayed ranking. The bot is just another player here; its own Blackjack settlements update its `user_wallet` and `casino_account` rows like any human player.
- Cumulative casino-system P&L lives in the `casino_ledger` row (`ledger_id="casino"`) in `data/global_state.db`, separate from the bot's own `user_wallet`. `apply_round_settlement` writes the player delta into `economy.db` and the inverse `casino_delta` into `casino_ledger`; cross-engine writes follow the existing jackpot split-file limitation, with ordinary pre-commit errors rolling both sessions back and the player wallet committed before the casino ledger so user balance wins if only one final commit succeeds. `/casino` reads this ledger; `/pocat` reads the bot's `user_wallet`. Do not reuse the bot's `user_wallet` as the house ledger.
- Economy visibility follows the event privacy boundary. Public embeds are for shared social, market, or settlement events that other players can react to: `/give`, leaderboards, `/casino`, `/pocat`, successful admin adjustments, loan requests and decisions, `/credit repay`, `/credit call`, `/central_bank status`, `/central_bank repay`, and `/central_bank call`. Public economy embeds schedule cleanup after send.
- Ephemeral economy embeds are for personal state and validation or permission failures: `/balance`, `/checkin`, `/vip`, `/credit status`, insufficient-balance failures, missing-loan failures, and permission errors.
- `cli.py` grants the global 5,000-point message reward for every non-bot message. `gen_reply.py` adds token-based chat reward only after streamed AI replies. Other cogs should not invent action rewards.

## Stocks

- Simulated stock state lives in `data/stock.db`; wallet cash remains in economy `user_wallet`. Do not store wallet balances in stock tables.
- `/stock` sends one public market message. Stock selection, balance, positions, actions, news, validation, settlement, position summaries, recent trade history, and the 7D chart all edit that same public message. Only the user who opened the stock message can operate its controls. The active stock view deletes its message after 180 idle seconds.
- `stock_profile` is the source of truth for virtual company symbols, names, categories, prices, liquidity, volatility, fair value, mean reversion, tick caps, and news cadence. Do not hardcode company rows or seed companies from runtime code; use `uv run python scripts/manage_stock_company.py ...` or direct DB maintenance while the bot is stopped. This repo assumes offline DB maintenance for stock schema changes; do not add legacy stock schema migrations.
- Stock settlement uses `float_shares` as the cap for aggregate long exposure and aggregate short borrow capacity. Opening new long also clamps each user to 49% of `float_shares`. Selling long or covering short is always allowed as risk-reducing flow; opening new short clamps to the remaining DB-managed borrow capacity.
- `stock_news` refreshes at most once per `news_cadence_hours` per symbol. `StockNewsAI` may use the existing `AsyncOpenAI` Responses API stack when LLM config is available, but database helpers must always have deterministic fallback news and must not block settlement on LLM failure.
- Stock news sentiment is impulse: each news contributes its `clamp(sentiment_bps, ±NEWS_SENTIMENT_LIMIT_BPS)` exactly once at its firing tick boundary inside `advance_market_in_session`, not as a decayed drift over every following tick. `_news_impulse_by_boundary` routes news that landed in skipped backlog ticks to the next surviving boundary. The decay-aware `_decayed_news_sentiment_for_context` is only fed to the AI news generation prompt as ambient sentiment.
- Stock news generation receives `StockNewsGenerationContext` with daily price change, recent decayed order-flow pressure, and existing news sentiment. Prompts and fallback copy templates live in `src/discordbot/cogs/_stock/prompts.py`; keep generated news fictional, absurd, harmless, and broadly aligned with that market context.
- For Discord UI that needs precise layout beyond embed and Markdown limits, render a PNG with Pillow, attach it with `File`, then reference it from the embed via `attachment://...`. Use a CJK-capable font such as Noto Sans CJK for Traditional Chinese, and keep source data typed so the image renderer stays presentation-only.
- Market pressure uses decayed order flow and per-stock liquidity. Do not reintroduce a multi-day aggregate pressure term that applies the same directional drift on every tick.
- Execution price uses order-size slippage from `liquidity_shares`, capped by `max_tick_change_bps`. Store and display the per-leg execution `price_cents`; do not settle large orders at the quote price.
- Stock tables that persist `user_id` also persist `user_name`. Public stock UI should display stored names instead of Discord IDs.
- Stock settlement must go through `settle_stock_operation(...)`. Views must not split price reads, wallet reads, and position writes or import SQLAlchemy stock models directly.
- Stock portfolio views have a short process cache invalidated by stock writes and profile upserts. Market board and 7D chart PNG caches are digest-keyed; include every pixel-affecting quote or tick field in the immutable render key instead of adding ad hoc stale state.
- Stock persisted money and share quantity columns use decimal string storage and are parsed to Python `int` for arithmetic so they do not inherit SQLite's 64-bit integer ceiling. Wallet deltas stay integer `CURRENCY_NAME`; use `cash_ceil(...)` and `cash_floor(...)` for conversion.
- Compound stock operations write one `stock_operation` and ordered `stock_trade_leg` rows. Do not net wallet legs before applying them through the public ordered economy helper.
- Use `uv run python scripts/manage_stock_reconciliation.py list` to inspect non-final stock operations before manual repair. Use `uv run python scripts/manage_stock_company.py audit` to inspect float supply, aggregate exposure, and remaining long / short capacity.
- Lazy market ticks advance every 5 minutes on interaction. Long backlogs compress to at most `MAX_TICKS_PER_INTERACTION` ticks and still roll over Asia/Taipei day boundaries.

## Games

- Pure rules live in `cogs/_games/blackjack.py` and `cogs/_games/dragon_gate.py`; production uses `random.SystemRandom`, tests inject seeded `random.Random`.
- Lobby scaffolding lives in `cogs/_games/lobby.py`. Do not convert the base views to `abc.ABC`; project style uses `raise NotImplementedError`.
- Casino settlement is one atomic step after the round resolves. Validate or clamp bets up front, then settle through the game settlement helpers.
- Blackjack supports Hit, Stand, Double Down, Split, Surrender, Insurance, and peek. `/games blackjack bet=0` means all in; the slash `bet` option is string input parsed by the bot so large wagers are not capped by Discord integer options. Split uses same Blackjack value, so 10/J/Q/K can split with each other. No Double after Split. Split-hand 21 is not natural Blackjack. Table capacity is `MAX_BLACKJACK_PLAYERS = 6` to leave room for the bot player alongside up to five humans.
- Blackjack settlement uses `BlackjackHandState` plus `BlackjackHandSettlement` / `BlackjackPlayerSettlement`. Do not reintroduce the legacy single-hand `BlackjackHand`, `settle()`, or `settle_blackjack_round()` path.
- A Blackjack hand that reaches five or more cards without busting auto-stands as 過五關. Non-21 過五關 pays a normal win regardless of dealer total. Five-or-more-card 21 keeps the existing main-hand settlement against the dealer plus the extra 1x system-funded bonus. VIP adds one 0.5x bonus for that qualifying 21 bonus, but `/casino` only mirrors dealer-paid normal settlement.
- The dealer is the casino system, not the bot. `BlackjackRound._play_dealer` is fully deterministic under H17 (≤16 hit, soft 17 hit, hard 17+ stand) and never calls an LLM. The `BlackjackDealerStep.source` Literal is therefore `"auto" | "guard"`.
- The bot account sits at every Blackjack table as a regular player. `games.py` auto-adds it during lobby creation when `bot_user.wallet > 0`; bet sizing, hit/stand/double/split/surrender, and insurance are all decided by `BotPlayerAI` (slow_model, 30s timeout, basic-strategy fallback). The bot settles through `settle_blackjack_player` like every human, so its wallet, daily counters, and leaderboard rows behave the same.
- `BotPlayerAI` receives `BotFinancialContext` (lifetime balance / earned / spent + today's casino loss / win / net) and `OtherPlayerView`s (every non-bot player's visible hands + bets) so its prompt can reason from real table state. Decision results carry a `reason` field that surfaces as `💭 ...` on the bot's seat embed. Prompts in `cogs/_games/prompts.py` describe Blackjack rules and prod the model to reason; do not turn them back into prescriptive basic-strategy lookup tables.
- Blackjack rounds deal from a shuffled multi-deck shoe (`build_shoe`, default `SHOE_DECK_COUNT = 4`) stored on `BlackjackRound.shoe`. All internal draws go through `BlackjackRound._draw_one_card`; the module-level `draw_card` stays as the empty-shoe fallback and test monkeypatch seam — tests that need deterministic draws clear `round_state.shoe` before monkeypatching.
- Blackjack action buttons are presence-based: invalid controls are removed from the view instead of being left visible and disabled.
- Blackjack Hit hints and final settlement banter are background refreshes. Player interactions and final result publication must not wait on LLM latency.
- Blackjack peek runs as a two-stage view animation when the dealer up-card is A or 10/J/Q/K, gated by `BlackjackView._peek_animated`. Do not collapse `_animate_peek_reveal_bj_locked` / `_animate_peek_no_bj_locked` into a single edit; players need the "莊家偷看 → 翻牌" beats.
- `BlackjackView._finalize_locked` disables current visible buttons and calls `self.stop()` before settlement. The final edit removes controls with `view=None`, and `interaction_check` replies with an ephemeral notice once `_settled=True`.
- Blackjack player losses clamp at balance 0. `apply_round_settlement` mirrors only the actual collected player debit into `casino_ledger`; full positive payouts still mirror the casino-paid delta.
- Dragon Gate is backed by the shared `jackpot_pool` row `game_id="dragon_gate"` in `data/global_state.db`. Jackpot money columns also use decimal-string storage. Do not route it through the casino ledger.
- Dragon Gate ante, losses, wins, leave refunds, and timeout refunds settle through jackpot helpers. Losses clamp at balance 0.
- On Dragon Gate leave or timeout, positive per-player running delta is refunded into the pool unless the whole-pool win branch already cleared the jackpot.
- Interactive game and lobby views use 180-second idle timeouts. Terminal public messages schedule deletion 180 seconds after settlement or send through `utils.message_cleanup`; do not add cog-local delete/forget loops.
- `build_system_talk_embed` is the dedicated narrator-talk embed. Game messages now send one embed per seat: in-progress / final tables go out as `[talk, dealer_seat, *player_seat_embeds]` (history embed optional for Dragon Gate). Use `build_in_progress_embeds` / `build_final_embeds` (return `list[Embed]`) plus `build_dealer_seat_embed` / `build_player_seat_embed` for the per-seat split.
- Discord markdown headings render reliably only inside `embed.description`. Put card, total, and result sections there; use fields for auxiliary details.
- Player seat embeds carry the participant avatar as a `set_thumbnail` (right corner). The dealer seat deliberately omits a thumbnail so the bot's avatar does not double up between its dealer mask and its player seat.
- The dealer seat color follows the casino-vs-table outcome at settlement, not the dealer hand total: any player win paints the seat red, all-loss paints it green, an all-push table paints it yellow.
- The "dealer" is a label, not a Discord identity. `SystemNarrator` produces neutral broadcast lines (`SYSTEM_*` prompts) and uses the fixed end-user-id label `"casino_taunt_bet" / "casino_settle" / "casino_table_settle" / "casino_hint"`. The message is still sent by the bot account so `log_msg.py` records the speaker correctly.
- `GamesCogs.narrator` and `GamesCogs.bot_player_ai` are `cached_property`s. Each cog with an LLM client owns its own client intentionally.

## Other Cogs

- `parse_threads.py` watches for `threads.net` and `threads.com` URLs and uses reactions for status. It adds no reward beyond the global message reward.
- `video.py` keeps progress text on the deferred original message, then edits that same message with the final file and source URL.
- `auto_unmute.py` clears timeouts applied to the bot, finds the moderator from recent audit log entries, and replies in the last active human channel or the guild system channel.
- `log_msg.py` logs human messages and this bot's own replies, never third-party bots. Streaming replies converge via UPSERT on `discord_message_id`.
- `log_msg.py` owns module-level `_sql_engine`; do not move it to a per-message cached property.

## Config And Types

- Environment-backed config classes use `pydantic_settings.BaseSettings` with explicit `validation_alias=AliasChoices("ENV_NAME")`. `.env` is loaded at import time.
- Pure shared result types, enums, and constants live under `src/discordbot/typings/` when they do not depend on cogs or utils.
- Use `pydantic.BaseModel` for structured data and type-managed payloads. Do not introduce `dataclass`, and do not replace real payload types with loose `object` / `Any` annotations.
- Keep `Field(description=..., examples=...)` populated for configurable values.

## Coding Conventions

- Ruff is formatter and linter. Use narrow `# noqa: <rule>` comments with a reason when needed.
- mypy runs in pre-commit. Type hints should use the real domain type; avoid `object`, `Any`, or similarly loose annotations for convenience.
- Keyword arguments are required for normal function calls, including single-argument calls. Do not add bare `*` to new function signatures solely to force keyword-only usage.
- Allowed positional idioms include `len(x)`, `str(x)`, `Path("x")`, exception constructors, variadic collectors, and `logfire.info("message")`.
- Avoid intermediate one-level aliases such as `usage = responses.usage` when direct access is clearer.
- LLM latency matters. Do not add extra LLM calls for cosmetic improvements.
- Comments should explain non-obvious behavior only.
- `docs/` is generated by `make gen-docs`; do not hand-edit it.
- Do not touch the README badge block unless the user explicitly asks for badge maintenance.

## Documentation Split

- `README.md` is the concise, canonical user-facing README.
- `README.zh-CN.md` and `README.zh-TW.md` mirror `README.md` structure.
- `CONTRIBUTING.md` is developer-facing and stays in English.
- `CLAUDE.md` is AI-agent-facing. Keep it dense and project-specific.

## Text Formatting

- Do not reflow human-written prose.
- Do not hard-wrap Markdown or documentation text to 72, 80, or 100 columns. Editors should handle visual wrapping.
- When modifying documents, make the smallest textual diff possible and preserve the surrounding line structure.
- A prose paragraph should usually stay on one logical line unless the existing file is intentionally and consistently manual-wrapped.
