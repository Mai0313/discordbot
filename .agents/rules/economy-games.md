---
description: "Read before changing, planning or reviewing the economy or the casino games: cli.py, cogs/economy/, cogs/games/, services/economy/, their typings, the amount-parsing, avatar, interaction-response, stored-integer and number-text utils, scripts/modify_balance.py, and their tests and helpers."
trigger: model_decision
paths:
  - "src/discordbot/cli.py"
  - "src/discordbot/cogs/economy/**"
  - "src/discordbot/cogs/games/**"
  - "src/discordbot/services/economy/**"
  - "src/discordbot/typings/economy.py"
  - "src/discordbot/typings/games.py"
  - "src/discordbot/utils/amount_parsing.py"
  - "src/discordbot/utils/avatars.py"
  - "src/discordbot/utils/interaction_responses.py"
  - "src/discordbot/utils/number_text.py"
  - "src/discordbot/utils/stored_integer.py"
  - "scripts/modify_balance.py"
  - "tests/test_cli.py"
  - "tests/cogs/economy/**"
  - "tests/cogs/games/**"
  - "tests/services/economy/**"
  - "tests/guards/test_ack_before_write.py"
  - "tests/scripts/test_modify_balance.py"
  - "tests/typings/test_economy.py"
  - "tests/utils/test_amount_parsing.py"
  - "tests/utils/test_avatars.py"
  - "tests/utils/test_interaction_responses.py"
  - "tests/utils/test_number_text.py"
  - "tests/helpers/dragon_gate.py"
  - "tests/helpers/economy.py"
  - "tests/helpers/economy_invariants.py"
  - "tests/helpers/games.py"
  - "tests/helpers/test_economy_invariants.py"
---

## Economy

- **Economy identity is cross-server: `UserAccount` has no `guild_id`.**
- **`total_earned - total_spent == balance` must hold:** every applied positive delta bumps `total_earned`, every negative one bumps `total_spent`.
- **Money inputs are string `SlashOption`s parsed by `utils.amount_parsing.parse_decimal_amount`;** malformed text gets an ephemeral reply before any mutation. Reuse it for any number that can exceed Discord's int cap.
- **`UserAccount.avatar_url` is a last-seen cache (`utils.avatars.guild_avatar_url`); do not backfill existing URLs.**
- **Central-bank minting is bounded by the pool and the per-borrower ceiling, never by the approver.** Approval is any server administrator (`cogs/economy/views.py::is_guild_admin`), which anyone becomes by creating a server.
- **The pool's collateral is per guild; its debt is the whole bank's.** `guild_participant` is recorded only for a rewarded message's author and a `/central_bank` caller, never a `member:` target.
- **Shared social and settlement events are public embeds with scheduled cleanup; personal state, malformed-amount and permission failures are ephemeral.** A validation failure found after the public defer (`/give`, `/games`) is a public expiring embed.
- **Exactly one faucet pays an action reward, `cli.py`'s cooldown-gated per-message reward;** other cogs must not add a second (the system-funded mints that settle a loan or a game are not rewards).
- **`apply_vip_blackjack_bonus` and the loan-rate converters stay in `typings/economy.py`,** so code that only formats or settles a number never imports the ledger's engine module (#610).

## Games

- `lobby.py`'s base views keep `raise NotImplementedError`; do not convert them to `abc.ABC`.
- **No LLM anywhere in the games:** do not reintroduce the removed casino `SystemNarrator` banter, the bot-player `reason` text or any AI bot-player. The dealer is a label, not a Discord identity, and posts no narrator messages.
- **Each Blackjack seat and each Dragon Gate bet settles in one atomic step once it resolves:** validate or clamp bets up front, then settle through the helpers.
- **A human player sits at one started casino round at a time, Blackjack and 射龍門 alike** (`cogs/games/seats.py` has why): a new game goes through the lobby's seats too. A Blackjack seat holds the most its round can take, so **a new path that debits a user's wallet of their own accord runs under `services/economy/holds.py::spending` and checks `held_amount` / `spendable`**, as every existing voluntary debit does.
- **Action buttons are presence-based: an invalid control is removed, not disabled.**
- **A settled table goes out through `games/interactions.py::publish_final_table`**, which schedules its deletion; never delete terminal public messages in a cog-local loop.
- **The blackjack bot player (`bot_player.py`, no LLM) may drive the casino ledger negative**: the owner asked for a strong bot, so house losses are not a bug to fix in it.
