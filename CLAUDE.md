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
- Cog loading is synchronous inside `DiscordBot.__init__` before gateway
    connection. First `on_ready` syncs global application commands and starts the
    status task.
- Every cog module must expose sync `def setup(bot): ...` and add cogs with
    `override=True`. Do not use `async def setup`; nextcord schedules it without
    awaiting, so the first command sync can see no commands.
- Helper packages live in sibling `_<cog>/` directories so they are not
    auto-loaded as cogs.
- Add common command errors in `DiscordBot.on_command_error` instead of catching
    them in each cog.

## Cog Rules

- Cog modules should define one `commands.Cog` subclass plus a sync `setup`.
- Cogs should not import peer cogs directly. Use the bot instance, shared
    typings, or cog-private helper packages.
- Slash commands need localized names and descriptions for English,
    Traditional Chinese, and Japanese where the command is user-facing.
- Any user-visible command or behavior change must update
    `src/discordbot/cogs/help.py` in the same change and keep
    `tests/test_help.py` passing.

## AI Pipeline

- Runtime LLM backend is LiteLLM Proxy behind `OPENAI_BASE_URL`; all runtime
    conversations use `AsyncOpenAI` and the OpenAI Responses API. Do not switch
    chat, routing, or captioning back to Chat Completions.
- Provider selection happens through LiteLLM via model strings,
    `ModelSettings.tools`, and `extra_body`.
- Do not import provider-native SDKs such as `google-genai` or `anthropic` into
    runtime request paths. `scripts/prompt_dev.py` is the exception for local
    experimentation.
- Runtime model strings live in `RuntimeModelCatalog` in
    `src/discordbot/typings/models.py`; update that catalog and keep
    `slow_model`'s peak-hour dispatch.
- Streaming SDK objects are named `responses`; loop items are named `response`.
- AI progress is communicated with reactions on the user's message. Preserve the
    no-intermediate-message UX.
- Attachment ingestion is gated by `get_supported_modalities` for the slow
    model. Unsupported attachments should be dropped before any LLM call.
- Images are resized and JPEG re-encoded into `input_image` data URIs. Other
    supported attachments are sent as `input_file`.
- For Discord embeds, prefer `media.discordapp.net` `proxy_url` over origin URLs
    because CDN links expire.
- History, referenced messages, and current messages are fetched with
    `asyncio.gather`. Keep the task list built in a `for` loop on its own lines.

### Responses API Gotchas

- Current OpenAI models are strict about role and content part pairing:
    `user` / `system` / `developer` use `input_text`, `input_image`, and
    `input_file`; `assistant` uses `output_text` or `refusal`.
- Use `EasyInputMessageParam` plus the concrete Responses content part types,
    then cast only at the SDK boundary.
- Prefer string-content shorthand when there are no attachments.
- Processed Discord messages with attachments fall back to `role=user` because
    assistant content cannot contain `input_image` or `input_file`.
- Separator messages use `role=system`, not `developer`, for Gemini and Claude
    compatibility through LiteLLM.
- Gemini may prepend leading newlines to streamed reasoning output when
    `reasoning.effort != "none"`. `_handle_streaming` strips them on the first
    content delta; keep that guard.
- Gemini thought summaries only flow through the Responses API.
- `client.images.edit(image=...)` needs raw `bytes`, not image-reference dicts.
- Per-token pricing comes from `utils.model_pricing` and its cached LiteLLM
    JSON. Do not hardcode rates in `gen_reply.py`.
- `message.snapshots` are intentionally not walked by cleaned-content or
    attachment ingestion.
- `BELIEF` in `_gen_reply/prompts.py` is currently disabled but kept in the
    signature for a possible future re-enable.

## Economy

- SQLite files are separate: `data/messages.db` for message logs,
    `data/economy.db` for 虛擬歡樂豆, and `data/game_cleanup.db` for cleanup
    targets.
- `cogs/_economy/database.py` owns the module-level async engine `_engine`.
    Do not move it to `cached_property`; tests monkeypatch `_engine` and expect
    helpers to bind sessions directly from the current object.
- `UserAccount` has no `guild_id`. Balances, VIP, loans, check-ins, admin flags,
    leaderboard visibility, and leaderboards are cross-server by design.
- `UserAccount.avatar_url` is a last-seen cache. Discord-facing write paths use
    `utils.avatars.guild_avatar_url(...)` with guild context so guild avatars are
    stored when available, falling back to global `display_avatar`. Do not add a
    migration for existing avatar URLs; they refresh naturally on later writes.
- Every balance mutation should write a `PointTransaction` row unless the helper
    intentionally skips `delta == 0`.
- `credit_with_repayment` is the income path for message reward, chat reward,
    and casino payout. The auto-repay slice is controlled by the
    `auto_repay_ratio_percent` inline constant in
    `_credit_with_repayment_in_session`; it is currently `0` (no diversion),
    and bumping it back to `50` restores the half-of-income-to-principal
    behavior without restructuring callers.
    `/give` recipients are not auto-repaid.
- Loan helpers reset stale principal lazily at Asia/Taipei midnight. Borrowing
    over the remaining daily cap clamps to the remaining cap; only zero remaining
    credit rejects.
- Admin adjustments use `adjust_balance(..., allow_negative=..., note=...)` and
    `MANUAL_ADJUSTMENT`, not casino transaction kinds.
- `UserAccount.hide_from_leaderboard` defaults to `False`. When set, `/leaderboard`
    and `/loss_leaderboard` omit that account; maintenance callers can still opt
    into hidden rows.
- Daily casino counters live on `user_account` as
    `casino_day_started_at`, `daily_casino_loss`, `daily_casino_win`, and
    `daily_casino_net`. Player-side Blackjack and Dragon Gate settlements
    update them from the actual applied delta; house ledger rows, push rounds,
    manual adjustments, transfers, loans, rewards, check-ins, and VIP purchases
    do not. Blackjack five-card 21 system bonuses count as player-side casino
    payout and daily win/net, but do not move `/house`. `/loss_leaderboard`
    reads gross `daily_casino_loss`, so wins do not offset the displayed
    ranking.
- `credit_limit(user, *, is_vip)` is pure and tiered by Discord account age.
    Keep the tier table inline in that function.
- `/balance`, `/borrow`, `/repay`, `/checkin`, `/vip`, and admin error replies
    are private. Public economy embeds schedule cleanup after send.
- `cli.py` grants the global 5,000-point message reward for every non-bot
    message. `gen_reply.py` adds token-based chat reward only after streamed AI
    replies. Other cogs should not invent action rewards.

## Games

- Pure rules live in `cogs/_games/blackjack.py` and
    `cogs/_games/dragon_gate.py`; production uses `random.SystemRandom`, tests
    inject seeded `random.Random`.
- Lobby scaffolding lives in `cogs/_games/lobby.py`. Do not convert the base
    views to `abc.ABC`; project style uses `raise NotImplementedError`.
- Casino settlement is one atomic step after the round resolves. Validate or
    clamp bets up front, then settle through the game settlement helpers.
- Blackjack supports Hit, Stand, Double Down, Split, Surrender, Insurance, and
    peek. Split uses same Blackjack value, so 10/J/Q/K can split with each
    other. No Double after Split. Split-hand 21 is not natural Blackjack.
- A Blackjack hand that reaches exactly five cards totaling 21 auto-stands as
    five-card 21. The main hand still settles against the dealer normally, and
    the extra 1x bet bonus is system-funded. VIP adds one 0.5x bonus for that
    qualifying hand, but `/house` only mirrors dealer-paid normal settlement.
- Blackjack dealer play is deterministic only below 17: ≤16 hits. At 17+
    the interactive dealer phase calls `DealerAI.decide_blackjack_action` and
    applies the returned hit / stand action.
- Blackjack action buttons are presence-based: invalid controls are removed
    from the view instead of being left visible and disabled.
- Blackjack Hit hints and final settlement banter are background refreshes.
    Player interactions and final result publication must not wait on LLM
    latency.
- Blackjack peek runs as a two-stage view animation when the dealer up-card
    is A or 10/J/Q/K, gated by `BlackjackView._peek_animated`. Do not collapse
    `_animate_peek_reveal_bj_locked` / `_animate_peek_no_bj_locked` into a single
    edit; players need the "莊家偷看 → 翻牌" beats.
- `BlackjackView._finalize_locked` disables current visible buttons and calls
    `self.stop()` before settlement. The final edit removes controls with
    `view=None`, and `interaction_check` replies with an ephemeral notice once
    `_settled=True`.
- Blackjack player losses clamp at balance 0. `apply_round_settlement` mirrors
    only the actual collected player debit into the bot's house ledger; full
    positive payouts still mirror the dealer-paid delta.
- Dragon Gate is backed by the shared `jackpot_pool` row
    `game_id="dragon_gate"`. Do not route it through the house ledger.
- Dragon Gate ante, losses, wins, leave refunds, and timeout refunds settle
    through jackpot helpers. Losses clamp at balance 0.
- On Dragon Gate leave or timeout, positive per-player running delta is refunded
    into the pool unless the whole-pool win branch already cleared the jackpot.
- Interactive game and lobby views use 180-second idle timeouts. Terminal public
    messages schedule deletion 180 seconds after settlement or send.
- `build_dealer_talk_embed` is the dedicated dealer-talk embed. In-progress and
    final game messages may send multiple embeds in order
    `[dealer talk, main, history?]`.
- Discord markdown headings render reliably only inside `embed.description`.
    Put card, total, and result sections there; use fields for auxiliary details.
- Dealer identity comes from `bot.user`. The message is still sent by the bot
    account so `log_msg.py` records the speaker correctly.
- `GamesCogs.dealer` is a `cached_property`. Each cog with an LLM client owns
    its own client intentionally.

## Other Cogs

- `parse_threads.py` watches for `threads.net` and `threads.com` URLs and uses
    reactions for status. It adds no reward beyond the global message reward.
- `video.py` keeps progress text on the deferred original message and sends the
    final file via followup. Do not collapse file delivery into
    `edit_original_message`; Discord may drop content on multipart edits.
- `auto_unmute.py` clears timeouts applied to the bot, finds the moderator from
    recent audit log entries, and replies in the last active human channel or the
    guild system channel.
- `log_msg.py` logs human messages and this bot's own replies, never third-party
    bots. Streaming replies converge via UPSERT on `discord_message_id`.
- `log_msg.py` owns module-level `_sql_engine`; do not move it to a
    per-message cached property.

## Config And Types

- Environment-backed config classes use `pydantic_settings.BaseSettings` with
    explicit `validation_alias=AliasChoices("ENV_NAME")`. `.env` is loaded at
    import time.
- Pure shared result types, enums, and constants live under
    `src/discordbot/typings/` when they do not depend on cogs or utils.
- Use `pydantic.BaseModel` for structured data and type-managed payloads. Do not
    introduce `dataclass`, and do not replace real payload types with loose
    `object` / `Any` annotations.
- Keep `Field(description=..., examples=...)` populated for configurable
    values.

## Coding Conventions

- Ruff is formatter and linter. Use narrow `# noqa: <rule>` comments with a
    reason when needed.
- mypy runs in pre-commit. Type hints should use the real domain type; avoid
    `object`, `Any`, or similarly loose annotations for convenience.
- Keyword arguments are required for normal function calls, including
    single-argument calls. Do not add bare `*` to new function signatures solely
    to force keyword-only usage.
- Allowed positional idioms include `len(x)`, `str(x)`, `Path("x")`, exception
    constructors, variadic collectors, and `logfire.info("message")`.
- Avoid intermediate one-level aliases such as `usage = responses.usage` when
    direct access is clearer.
- LLM latency matters. Do not add extra LLM calls for cosmetic improvements.
- Comments should explain non-obvious behavior only.
- `docs/` is generated by `make gen-docs`; do not hand-edit it.
- Do not touch the README badge block unless the user explicitly asks for badge
    maintenance.

## Documentation Split

- `README.md` is the concise, canonical user-facing README.
- `README.zh-CN.md` and `README.zh-TW.md` mirror `README.md` structure.
- `CONTRIBUTING.md` is developer-facing and stays in English.
- `CLAUDE.md` is AI-agent-facing. Keep it dense and project-specific.
