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
- Cog loading is synchronous inside `DiscordBot.__init__` before gateway connection. First `on_ready` syncs global application commands, starts the status task, and warms the model-pricing cache off the event loop.
- The status task rotates presence lines from the `bot_status` table in `data/global_state.db` (managed offline via `scripts/manage_bot_status.py`); an empty table falls back to a built-in default.
- Every cog module must expose sync `def setup(bot): ...` and add cogs with `override=True`. Do not use `async def setup`; nextcord schedules it without awaiting, so the first command sync can see no commands.
- Helper packages live in sibling `_<cog>/` directories so they are not auto-loaded as cogs.
- Add common command errors in `DiscordBot.on_command_error` instead of catching them in each cog.

## Cog Rules

- Cog modules should define one `commands.Cog` subclass plus a sync `setup`.
- Cogs should not import peer cogs directly. Use the bot instance, shared typings, or cog-private helper packages.
- Slash commands need localized names and descriptions for English, Traditional Chinese, and Japanese where the command is user-facing.
- Any user-visible command or behavior change must update `src/discordbot/cogs/help.py` in the same change and keep `tests/test_help.py` passing.
- When one Discord message sends multiple embeds that need aligned widths, use `discordbot.utils.discord_embeds.embed_spacer_payload(...)`; pass `target=` so edits can retain an already-uploaded spacer by id instead of re-uploading it. Re-uploading the unchanging spacer on every edit trips Discord's per-message edit attachment upload limit (error code 400009) on rapidly edited messages like the Blackjack table. Genuinely changing PNGs still upload via `extra_files`.
- Discord embed hard limits (the API rejects the whole message on overflow): title 256 chars, description 4,096 chars, each field name 256 chars, each field value 1,024 chars, footer text 2,048 chars, author name 256 chars, at most 25 fields per embed, at most 10 embeds per message, and at most 6,000 characters summed across every embed in one message. Stay inside these when building embeds; when content cannot fit, paginate or render a PNG with Pillow and reference it via `attachment://...` instead of truncating silently.

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

- SQLite files are separate: `data/messages.db` for message logs, `data/economy.db` for per-user 虛擬歡樂豆, `data/global_state.db` for bot-wide shared state such as jackpot pools and bot presence rotation, and `data/game_cleanup.db` for public message cleanup targets.
- `cogs/_economy/database.py` owns the module-level async engines `_engine` and `_global_state_engine`. Do not move them to `cached_property`; tests monkeypatch those engines and expect helpers to bind sessions directly from the current object.
- `UserAccount` has no `guild_id`. Identity, VIP, check-ins, admin flags, central banker flags, and leaderboard visibility are cross-server by design. Spendable balances and gross totals live in `user_wallet`, which also denormalizes `name` for direct DB inspection; long-term loans live in `loan_proposal` and `loan_contract`; daily casino counters live in `casino_account`. Economy money columns are stored as decimal strings and parsed to Python `int` for arithmetic so they do not inherit SQLite's 64-bit integer ceiling.
- User-facing money amount inputs are string `SlashOption`s parsed by the bot, not integer options, so transfers, loans, and collections are not capped by Discord's integer option ceiling. `_parse_positive_amount` handles required positive amounts (comma-formatted, rejects 0/negative/non-decimal); `_parse_collect_amount` handles optional collection amounts where blank or `0` means collect-all. Malformed text returns `build_invalid_amount_embed` ephemerally before any mutation. Prefer this string-parse-at-the-boundary pattern for any new numeric input that can exceed Discord limits, converting to `int`/`float` only at compute time.
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
- `cli.py` grants the global `BASE_MESSAGE_REWARD_AMOUNT` message reward per non-bot message, gated by a process-local per-user `MESSAGE_REWARD_COOLDOWN_SECONDS` cooldown (`DiscordBot._message_reward_at`, resets on restart) so it cannot be spam-farmed. `gen_reply.py` adds a token-based chat reward after streamed AI replies, floored by `CHAT_REWARD_TOKEN_DIVISOR` and capped at `CHAT_REWARD_MAX_PER_REPLY`. Other cogs should not invent action rewards.
- Anti-inflation guardrails (all constants in `typings/economy.py`): faucets are deflated; `transfer()` burns `TRANSFER_TAX_BPS` of every `/give` as a permanent sink (sender debited full amount, receiver credited the net, the difference vanishes, per-side invariant preserved); every casino wager is capped at `MAX_SINGLE_BET` at the shared chokepoints (`_games/wagers.py::build_wager_participant` for Blackjack, `DragonGateRound.current_max_bet` for 射龍門) so balances cannot compound through all-in doubling; the VIP casino multiplier is `_VIP_WIN_MULTIPLIER_NUM/DEN` = 6/5 (1.2x).
- `scripts/reset_economy.py` is the offline reset (bot stopped, back up the `.db` files first; always `--dry-run` against a copy). It routes through `_economy/database.py` helpers — `reset_all_wallets` (log-compress / fixed / wipe; sets the full `(balance, total_earned, total_spent)` triple so the invariant holds), `set_wallet_exact`, `reset_casino_daily_counters`, `reset_casino_ledger`, `reset_jackpot_pools`, `forgive_loan_contracts`, `expire_loan_proposals`, `count_wallet_invariant_violations` — plus `_stock/database.py::reset_all_positions`. All write through the ORM so StoredInteger text stays canonical; cross-engine targets use their own session.

## Stocks

- Simulated stock state lives in `data/stock.db`; wallet cash remains in economy `user_wallet`. Do not store wallet balances in stock tables.
- `/stock` sends one public market message. Stock selection, balance, positions, actions, news, validation, settlement, position summaries, recent trade history, and the 7D chart all edit that same public message. Only the user who opened the stock message can operate its controls. The active stock view deletes its message after 180 idle seconds.
- `stock_profile` is the source of truth for virtual company symbols, names, categories, prices, liquidity, volatility, fair value, mean reversion, tick caps, and news cadence. Do not hardcode company rows or seed companies from runtime code; use `uv run python scripts/manage_stock_company.py ...` or direct DB maintenance while the bot is stopped. This repo assumes offline DB maintenance for stock schema changes; do not add legacy stock schema migrations.
- Stock settlement uses `float_shares` as the cap for aggregate long exposure and aggregate short borrow capacity. Opening new long also clamps each user to 49% of `float_shares`. Selling long or covering short is always allowed as risk-reducing flow; opening new short clamps to the remaining DB-managed borrow capacity.
- `stock_news` refreshes at most once per `news_cadence_hours` per symbol. `StockNewsAI` may use the existing `AsyncOpenAI` Responses API stack when LLM config is available, but database helpers must always have deterministic fallback news and must not block settlement on LLM failure.
- Stock news sentiment is impulse: each news contributes its `clamp(sentiment_bps, ±NEWS_SENTIMENT_LIMIT_BPS)` exactly once at its firing tick boundary inside `advance_market_in_session`, not as a decayed drift over every following tick. `_news_impulse_by_boundary` routes news that landed in skipped backlog ticks to the next surviving boundary. The decay-aware `_decayed_news_sentiment_for_context` is only fed to the AI news generation prompt as ambient sentiment.
- Stock news generation receives `StockNewsGenerationContext` with daily price change, recent decayed order-flow pressure, and existing news sentiment. Prompts and fallback copy templates live in `src/discordbot/cogs/_stock/prompts.py`; keep generated news fictional, absurd, harmless, and broadly aligned with that market context.
- For Discord UI that needs precise layout beyond embed and Markdown limits, render a PNG with Pillow, attach it with `File`, then reference it from the embed via `attachment://...`. Use a CJK-capable font such as Noto Sans CJK for Traditional Chinese, and keep source data typed so the image renderer stays presentation-only. Shared font loading and text-anchoring helpers live in `utils/pil_text.py`; reuse them instead of re-implementing per renderer.
- Market pressure uses decayed order flow and per-stock liquidity. Do not reintroduce a multi-day aggregate pressure term that applies the same directional drift on every tick.
- Anti-inflation / realism guardrails on the price formula live as constants in `cogs/_stock/market.py` and apply globally regardless of per-company DB tuning: `MARKET_VOLATILITY_SCALE_BPS` scales the per-tick random width down toward real-market magnitudes via `effective_volatility_width_bps`, `GLOBAL_MAX_TICK_CHANGE_BPS` hard-caps every tick under the per-company `max_tick_change_bps`, and `apply_daily_price_limit` enforces a Taiwan-style `DAILY_PRICE_LIMIT_BPS` band against `previous_close_price_cents` inside `advance_market_in_session` (set the new close first on day rollover, then clamp). Re-measure these (and `PRESSURE_LIMIT_BPS`) with `uv run python scripts/simulate_stock_market.py`, which reports realized daily volatility, daily-limit hit rate, and terminal drift vs fair value. Reducing per-tick swing and capping the daily move shrinks the volatility-harvesting and buy-pressure pump loops that mint money.
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
- A Blackjack hand that reaches five or more cards without busting auto-stands as 過五關. Non-21 過五關 pays a normal win regardless of dealer total. Five-or-more-card 21 keeps the existing main-hand settlement against the dealer plus the extra 1x system-funded bonus. For a VIP, the round's VIP bonus is `max(0.2x of the dealer-paid win, 0.2x of that five-card 21 bonus)` — the larger single 0.2x bonus, not the sum of both; `/casino` only mirrors dealer-paid normal settlement.
- The dealer is the casino system, not the bot. `BlackjackRound._play_dealer` is fully deterministic under H17 (≤16 hit, soft 17 hit, hard 17+ stand) and never calls an LLM. The `BlackjackDealerStep.source` Literal is therefore `"auto" | "guard"`.
- The bot account sits at every Blackjack table as a regular player. `games.py` auto-adds it during lobby creation when `bot_user.wallet > 0`. Decisions are deterministic, not LLM-chosen: bet sizing is fractional-Kelly (`kelly_bet`), action is the EV engine's hole-aware `recommended_action` (`choose_bot_action`), and insurance is the count-based recommendation (`fallback_insurance`); `BotPlayerAI` only narrates the `reason` text off the critical path. The bot settles through `settle_blackjack_player` like every human, so its wallet, daily counters, and leaderboard rows behave the same.
- `BotPlayerAI` receives `BotFinancialContext` (lifetime balance / earned / spent + today's casino loss / win / net), `OtherPlayerView`s, and computed Blackjack context. The model-facing context exposes only the dealer up-card (`DealerKnowledge` is up-card-only); the hole card, the combined two-card dealer total, and `natural_blackjack` are never sent to the LLM, and the bot's `reason` cannot reference them. `cogs/_games/blackjack_ev.py` is a pure, no-LLM engine that runs two passes over H17 rules and this table's payouts (including the five-card-21 bonus): an exact, hole-aware pass picks `recommended_action` only (the bot's private edge), and a marginal pass integrates a hypothetical hole out over the remaining shoe only (the real hole is never added back, peek-conditioned on no Blackjack under an A/ten up-card) to produce the `dealer_outcome` distribution and per-action EVs that are the only numbers shown to the model, so for a given up-card and shoe they are identical regardless of the real hole; `recommended_expected_value` is that action's marginal EV. `fallback_action` (with `dealer_cards`/`shoe`) stays hole-aware via the exact pass; insurance is priced from the remaining-shoe ten-value density (`build_bot_insurance_context`, which is never even given the hole, take iff P(ten) > 1/3) so the bot cannot win insurance on a real dealer Blackjack in a normal shoe, and `fallback_insurance` reads that count-based recommendation. Split EV is a flagged estimate. Math decides, AI narrates: `choose_bot_action` plays `recommended_action` (hole-aware) deterministically and `fallback_insurance` (take iff P(ten) > 1/3) is the insurance decision; the LLM never picks the action or insurance. The deterministic template reason (`action_decision_reason` / `insurance_decision_reason`) shows on the `💭 ...` seat embed immediately, and `BotPlayerAI.narrate_bot_action_reason` / `narrate_bot_insurance_reason` upgrade it via a background `_refresh_bot_reason_later` (revision/settled-guarded, dropped if the table advanced). Prompts in `cogs/_games/prompts.py` are English narration prompts (the choice is fixed in `chosen_action` / `chosen_decision`) requiring a Traditional Chinese `reason`; do not turn them back into decision-making or prescriptive basic-strategy lookup tables.
- Bot bet sizing is deterministic fractional-Kelly in `kelly_bet`: half-Kelly via `BOT_KELLY_FRACTION`, hard-capped at `BOT_MAX_BET_FRACTION` of bankroll as a true ceiling (the owner-chosen table stake floors the bet only up to that ceiling, never past it, so a big table bet can't drag the bot over its risk limit), never LLM guesswork. The edge is `count_adjusted_edge(true_count)` = `BOT_TABLE_EDGE` + `BOT_EDGE_PER_TRUE_COUNT` × the channel's pre-deal Hi-Lo true count, so the bot spreads its wager by the count from the persistent shoe; with a fresh shoe the count is 0 and the edge is just `BOT_TABLE_EDGE`. Re-measure all three constants with `uv run python scripts/simulate_bot_blackjack.py` (neutral-count edge / variance) and its `--persistent` mode (edge-vs-true-count slope); the Monte Carlo also confirms the table is strongly +EV for the hole-aware bot.
- Blackjack rounds deal from a shuffled multi-deck shoe (`build_shoe`, default `SHOE_DECK_COUNT = 4`) stored on `BlackjackRound.shoe`. The shoe is persistent per Discord channel: `cogs/_games/shoe.py::BlackjackShoeStore` (in-memory on `GamesCogs`, dropped on restart) carries the depleted shoe across rounds so card counting has signal and the EV engine reasons over the real composition. `BlackjackLobbyView._start_game` takes the channel shoe (`take_shoe`, rebuilding past `RESHUFFLE_THRESHOLD_CARDS` and announcing a genuine reshuffle), passes it to `from_participants(shoe=...)`, and `BlackjackView._finalize_locked` saves the remaining `round_state.shoe` back (`save_shoe`). `compute_true_count` (Hi-Lo, in `blackjack_ev.py`) is the count helper. All internal draws still go through `BlackjackRound._draw_one_card`; the module-level `draw_card` stays as the empty-shoe fallback and test monkeypatch seam — tests that need deterministic draws clear `round_state.shoe` before monkeypatching.
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

## Fishing

- `/games fishing` is a single-player money sink + collection game living under `cogs/_fishing/`. State is in `data/fishing.db`; wallet cash stays in economy `user_wallet`. The cog is LLM-free; do not add an LLM client.
- It is one public message edited in place (panel / shop / cast reveal / leaderboard / stats), only the opener operates it, and it self-deletes after 180s idle. Mirror `_stock/views.py`: `FishingPublicView` + the central `edit_fishing_message(...)`; navigation builds a new view, stops the old one, and rebinds the message.
- `fish_grade_config`, `fish_species`, and `fishing_gear` are the tunable source of truth (grade weights, per-species intra-grade weight + base value + size range, rod/bait price + `rarity_shift_bps` + rod `durability` + bait `value_bonus_bps`). Seed offline with `uv run python scripts/manage_fishing.py seed-defaults`; do not seed from runtime and do not add stock-style migrations. `_fishing/defaults.py::build_default_catalog` is the one catalog definition, consumed only by the seed script, the simulator, and tests.
- `_fishing/catch.py` is pure and RNG-injected (`compose_grade_weights`, `roll_catch`): production passes `SystemRandom`, tests pass a seeded `random.Random`. Luck is the additive `rod.rarity_shift_bps + bait.rarity_shift_bps`, reweighting each grade by its rarity rank (`order_index`) and clamped to `[LUCK_FACTOR_MIN_BPS, LUCK_FACTOR_MAX_BPS]`; the most common grade is never moved.
- Net-deflationary by design: every cast's expected value is below the bait + amortized-rod cost, and each catch is capped at `FISHING_MAX_SINGLE_CATCH`. Re-measure after any catalog change with `uv run python scripts/simulate_fishing.py`, which must report every rod+bait combo as a sink. Purchases burn via `apply_ordered_wallet_deltas` (negative leg, no recipient); catch payout credits via `credit_with_repayment` (the casino-payout income path). This is a net sink like the casino house edge, not a new faucet.
- Cross-DB ordering: `purchase_gear` debits (burns) the wallet first, then grants gear in fishing.db, refunding on a grant failure. `settle_cast` consumes bait + durability and writes `catch_log` in fishing.db first, then credits the payout in economy.db; a payout that fails after the catch is logged returns `CastStatus.PAYOUT_DEFERRED` (never rolled back, only ever more deflationary). Hard crashes between the two file commits share the accepted non-atomicity of casino / jackpot settlement. No heavyweight operation-lifecycle table.
- Settlement goes through `settle_cast` and `purchase_gear`; views must not split price/wallet/state reads and writes or import the ORM models directly. Money and quantity columns use `StoredInteger`. The leaderboard is integer-aware DESC over `catch_log.value` (top single catches). Rods are durable and break at 0 durability; buying a rod replaces the current one; bait stacks per type. `catch_log` denormalizes `species_name`/`emoji`/`grade` so retuning the catalog never rewrites history.
- Fish are emoji for now. The big-emoji line in the reveal embed is the PNG seam: render with `utils/pil_text.py` and reference via `attachment://...` later without changing the layout. `scripts/reset_economy.py --reset-fishing` clears all per-user fishing state while leaving the catalog intact.

## Memory

- Per-user long-term memory lives as plain markdown files under `data/memories/` (gitignored): `<user_id>.md` is the consolidated main memory, `<user_id>.raw.md` accumulates phase-1 extractions. File-based on purpose (mirrors codex's memory folder); do not move it into SQLite. User id is the key; memory is cross-server and shared with DMs by design, like `UserAccount`.
- Two-phase pipeline modeled on codex `codex-rs/memories`, all in `cogs/_memory/`: after each streamed QA/SUMMARY reply, `pipeline.schedule_memory_update` (fire-and-forget, per-user in-flight de-dupe) runs phase-1 extraction over the already-built `message_list` + the `stream()` return value; once `RAW_CONSOLIDATION_THRESHOLD` entries (or byte cap) accumulate, phase-2 consolidation rewrites the main file and clears raw. Both phases use `RuntimeModelCatalog.memories_model` through `MemoryExtractorAI` (structured `responses.parse` + `asyncio.timeout` + return-None fallback). Memory must never add latency to the reply path: no awaiting the pipeline, and LLM failures silently keep the previous state.
- Phase-1 has a no-op gate (`has_signal=false` writes nothing); prompts enforce target-user-only extraction, user-message-over-assistant evidence, attribution phrasing, and treat conversation content as data, not instructions. The transcript renders each message as a column-0 `[message <n> | <role>]` marker with indented content, so bodies and display names cannot forge block boundaries or author prefixes at content start; keep that structure and the matching prompt rules together. `redact_secrets` (shape-specific patterns on purpose; a bare-hex rule would eat git SHAs) runs on the transcript before upload and on every model output. Keep these properties when editing `_memory/prompts.py`.
- The read path injects only the trigger user's main file, appended to the reply `instructions` via `render_memory_injection` (framed as background reference, not user instructions; embedded `=========` delimiter lookalikes are squashed so stored memory cannot fake an early end of the block). The main file must start with `v1` and stay under `MAIN_FILE_MAX_CHARS` so `/memory show` fits one embed description. `MemoryConfig` (`MEMORY_ENABLED`) is the kill switch for both paths.
- All file writes go through `_memory/store.py` under `store.user_lock(user_id)`; main-file writes are atomic (tmp + `os.replace`). `/memory clear` records `mark_cleared`, and in-flight updates check `cleared_since` before writing so a clear is not resurrected by a slower background task. Tests isolate state with the `memory_isolated_dir` fixture (monkeypatches `_MEMORY_DIR` and the process-local registries).

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
