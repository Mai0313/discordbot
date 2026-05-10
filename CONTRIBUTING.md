# Contributing

Thanks for your interest in contributing! This guide covers everything you need to set up a development environment and submit changes.

## Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- `ffmpeg` (for video stream merging)

### Getting Started

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot

# Install all dependencies (including dev, test, and docs groups)
uv sync --all-groups

# Set up environment
cp .env.example .env
# Edit .env with your tokens and API keys

# Run the bot
uv run discordbot
```

### Optional: MapleStory Artale Data

```bash
uv run python scripts/artale_data.py
```

Scrapes `artalemaplestory.com` and writes JSON files into `data/maplestory/`.

## Project Structure

```
src/discordbot/
├── __init__.py              # setup_logging() — configures logfire and tees stdout to ./data/logs/<timestamp>.log
├── cli.py                   # Main bot entry point (DiscordBot class)
├── cogs/                    # Command modules (auto-loaded, excluding __-prefixed files)
│   ├── auto_unmute.py       # Detects when the bot itself is timed out, clears the timeout, posts an AI reply
│   ├── _auto_unmute/
│   │   └── prompts.py       # UNMUTE_PROMPT
│   ├── economy.py           # /balance, /leaderboard, /give slash commands
│   ├── _economy/
│   │   └── database.py      # Per-user point balance store (SQLite) — async wrappers around SQLAlchemy ORM
│   ├── games.py             # /dice and /blackjack slash commands (single-player vs AI dealer)
│   ├── _games/
│   │   ├── blackjack.py     # Pure Blackjack rules: BlackjackHand, hand_value, settle, render_hand
│   │   ├── dealer.py        # DealerAI — fast-model wrapper for taunt_bet / settle / hint banter
│   │   ├── dice.py          # play_dice helper + render_rolls
│   │   ├── prompts.py       # DEALER_* prompts
│   │   └── views.py         # BlackjackView (Hit / Stand buttons) + embed builders
│   ├── gen_reply.py         # AI chat — @mention/DM trigger, routing, streaming via OpenAI Responses API
│   ├── _gen_reply/
│   │   ├── exceptions.py    # extract_friendly_error() — pulls the readable text out of nested LiteLLM/OpenAI errors
│   │   ├── prompts.py       # REPLY / ROUTE / SUMMARY / IMAGE / BELIEF / PERSONA prompts
│   │   └── views.py         # RegenerateView — single-button view for re-running an AI reply
│   ├── help.py              # /help slash command (localized guide)
│   ├── log_msg.py           # Message logging to SQLite
│   ├── maplestory.py        # /maple_* slash commands (8 commands)
│   ├── _maplestory/
│   │   ├── constants.py     # Display templates for stats
│   │   ├── embeds.py        # Discord embed builders
│   │   ├── models.py        # Pydantic data models
│   │   ├── service.py       # Data loading, search logic, caching
│   │   └── views.py         # Interactive UI components (dropdown select)
│   ├── parse_threads.py     # Threads.net auto-parser; awards points to the user who shared the link
│   ├── template.py          # /ping and utility reactions
│   └── video.py             # /download_video slash command; awards points on successful download (file delivered as a separate followup so the reward suffix isn't dropped)
├── typings/                 # Pydantic configuration & shared models
│   ├── config.py            # DiscordConfig (DISCORD_BOT_TOKEN, DISCORD_TEST_SERVER_ID)
│   ├── llm.py               # LLMConfig (OPENAI_BASE_URL, OPENAI_API_KEY)
│   └── models.py            # ModelSettings (name, effort, reasoning, tools) and RouteDecision
└── utils/
    ├── downloader.py        # yt-dlp video downloader wrapper
    ├── images.py            # Image URL / data URI conversion helpers
    ├── model_pricing.py     # LiteLLM price-table cache; get_token_rates() and get_supported_modalities()
    └── threads.py           # Threads.net content scraper

scripts/
├── artale_data.py           # Scrape Artale MapleStory data from artalemaplestory.com
├── gen_docs.py              # Generate mkdocstrings reference pages
├── gpt.py                   # Azure GPT-5.4 sandbox comparing chat.completions vs responses API
├── prompt_dev.py            # Prompt iteration / evaluation sandbox (OpenAI / Gemini / Anthropic SDK)
├── route_dev.py             # Route-classifier sandbox — client.responses.parse + Pydantic RouteDecision
├── test_fallback.py         # Sandbox for testing Litellm fallback behavior
└── video_dev.py             # Ad-hoc yt-dlp experiments

data/
├── logs/                    # Per-run log files written by setup_logging() (`<timestamp>.log`)
├── maplestory/              # MapleStory Artale game database
│   ├── monsters.json
│   ├── equipment.json
│   ├── scrolls.json
│   ├── npcs.json
│   ├── quests.json
│   ├── maps.json
│   ├── translations.json
│   ├── misc.json
│   └── useable.json
├── downloads/               # Temporary video download storage
├── threads/                 # Downloaded Threads.net media
├── messages.db              # SQLite message log written by cogs/log_msg.py
├── economy.db               # SQLite point-balance store (cross-server, no guild_id) written by cogs/_economy/database.py
└── model_prices.json        # Cached LiteLLM price table fetched by utils/model_pricing.py
```

### Architecture

- **Cog-based**: Each feature is a separate cog in `cogs/`. The bot auto-discovers and loads all `.py` files in the directory (excluding `__` prefixed files). Helper packages live in sibling `_<cog>/` folders so they are not auto-loaded.
- **Async**: Built on nextcord with async/await patterns throughout.
- **Config**: Pydantic models + `pydantic-settings` load from `.env` automatically (`DiscordConfig` in `typings/config.py`, `LLMConfig` in `typings/llm.py`). Shared model abstractions like `ModelSettings` and `RouteDecision` live in `typings/models.py`.
- **Logging**: `setup_logging()` in `discordbot/__init__.py` configures `logfire` (local console only, `send_to_logfire=False`) and tees stdout to `./data/logs/<timestamp>.log` for each run. `nextcord.state` logs are forwarded into logfire too.
- **LLM client**: Each cog that talks to the model owns a `cached_property AsyncOpenAI` client (`base_url=OPENAI_BASE_URL`, `api_key=OPENAI_API_KEY`) — currently `gen_reply`, `auto_unmute`, and `games` (whose `DealerAI` reuses the cog's client for dealer banter). The endpoint is OpenAI-compatible, typically a [Litellm](https://github.com/BerriAI/litellm) proxy fronting Gemini / Claude / OpenAI / etc., so model swaps are just a string change.
- **Economy**: Per-user point balances live in a separate SQLite (`data/economy.db`) managed by `cogs/_economy/database.py`. The schema is keyed by Discord `user_id` only — **no `guild_id`**, so balances and the `/leaderboard` are intentionally cross-server. Sync ORM helpers (`_add_balance_sync`, `_settle_game_sync`, `_transfer_sync`, `_top_n_sync`) are wrapped with `asyncio.to_thread` to keep the event loop responsive.
- **Game flow**: `/dice` and `/blackjack` (in `cogs/games.py`) deduct the bet up-front via `settle_game(delta=-bet)`, run pure-rule helpers from `cogs/_games/{dice,blackjack}.py`, then credit `bet + delta` back when the round resolves. `BlackjackView` (`cogs/_games/views.py`) drives Hit/Stand buttons and auto-stands on timeout. `DealerAI` (`cogs/_games/dealer.py`) wraps the fast model for `taunt_bet` / `settle` / `hint` banter; every entry point falls back to a hard-coded line on LLM failure so the round always resolves.
- **Slash command sync**: All slash commands are global (no `guild_ids`, no `force_global`). Registration goes through `sync_all_application_commands()` once on the first `on_ready`. Cogs are loaded synchronously in `DiscordBot.__init__` so application commands are populated before the gateway connects, and every cog's `setup` is `def setup(bot)` (sync) — `async def setup` would be fire-and-forgotten by `load_extension` and the first sync would see zero commands.
- **Model abstraction**: Models are not raw strings. `ModelSettings(name, effort)` (in `typings/models.py`) bundles the model identifier, reasoning effort, and the right `tools` shape per provider (Gemini `googleSearch` + `urlContext`, Claude `web_search_*` + `web_fetch_*`, others OpenAI `web_search`). Accepted input modalities are looked up separately via `get_supported_modalities()` from `utils/model_pricing.py` so `typings/` stays free of `utils/` imports. `gen_reply` exposes `fast_model` / `slow_model` / `image_model` / `video_model` as properties; `slow_model` is time-of-day dispatched (UTC weekdays 09:00–17:00 falls back to a lite model to avoid the Gemini Pro overload window).
- **AI Routing**: The `gen_reply` cog uses the fast model to classify user intent (`IMAGE` / `VIDEO` / `QA` / `SUMMARY`) via `client.responses.parse(text_format=RouteDecision)` and dispatches to the matching handler. All chat / route / caption calls use the **OpenAI Responses API** (not Chat Completions). The slow reply path streams the answer event-by-event (`response.output_text.delta`), strips Gemini's leading `\n\n\n` quirk, and ends with a Discord-quoted footer (`> **{model}** ⬆ in ⬇ out $cost`) where the cost comes from `discordbot.utils.model_pricing.get_token_rates` (a lazy fetch of the upstream LiteLLM `model_prices_and_context_window.json`, cached at `data/model_prices.json`). Processing progress is shown via emoji reactions on the user's message (🤔 → 🔀 → 🎨/🎬/📖/❓ → 🆗, plus 🌐 if web search fired, or ❌ on error).
- **Trigger rule**: In DMs the bot always responds; in guilds it only responds when the message text contains `<@bot_id>` (a reply-notification alone is ignored, so users replying to a Threads embed or a download result won't accidentally summon the bot).
- **Responses API input shape**: Build request messages with `EasyInputMessageParam` and `ResponseInputTextParam` / `ResponseInputImageParam` / `ResponseInputFileParam`, then cast to `ResponseInputParam` only at the `client.responses.*` call. Text-only bot messages may keep `role=assistant` with string content; any message that carries image or file parts must use `role=user` because assistant content cannot carry `input_image` or `input_file`. Separator / header messages (`==== Chat History ====` etc.) use `role=system` (compatible with Gemini/Claude via LiteLLM); the real system prompt is delivered through `instructions=`.
- **Attachment ingestion**: `_get_attachment_parts` first gates each attachment against `get_supported_modalities(model_name=slow_model.name)` — anything the slow model can't accept (e.g. video on a text-only model) is dropped before any LLM call. Surviving attachments are routed by `content_type`: `image/*` → PIL resize + JPEG re-encode → `input_image`; everything else (`video/*`, `application/pdf`, `text/plain`, etc.) → raw bytes → base64 data URI → `input_file`. Stickers and embed images/thumbnails (preferring `media.discordapp.net` proxy URLs) always go through the image path.
- **Auto-unmute**: A separate cog (`auto_unmute.py`) watches `on_member_update` for the bot itself transitioning into a future-dated `communication_disabled_until`, walks the recent audit log to find the moderator, clears the timeout via `member.edit(timeout=None, …)`, and posts a single AI sass-reply through its own `AsyncOpenAI` client. Reply target is the last channel a human spoke in (per guild), with `system_channel` as a fallback — Discord's `member_update` audit entry doesn't carry a channel.

## Code Standards

### Tooling

| Tool           | Purpose                                                                                                                      |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Ruff**       | Linting and formatting (line length: 99)                                                                                     |
| **mypy**       | Type checking with the Pydantic plugin (runs in pre-commit)                                                                  |
| **ty**         | Astral's type checker — installed as a dev dep, **not** in pre-commit. Run `uvx ty check src` manually for a second opinion. |
| **pre-commit** | Runs Ruff, mypy, ShellCheck, mdformat, codespell, gitleaks, etc. on commit                                                   |

### Style

- Follow PEP 8 naming conventions
- Use type hints on all functions
- Google-style docstrings
- Max line length: 99 characters
- Use Pydantic models for data validation

### Pre-commit Setup

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run all hooks manually
uv run pre-commit run -a
```

Hooks include: Ruff (check + format), mypy, ShellCheck, mdformat, codespell, gitleaks, nbstripout, uv-sync, uv-lock, plus standard hygiene hooks (`check-yaml`, `check-toml`, `detect-private-key`, `end-of-file-fixer`, `trailing-whitespace`, …).

## Testing

```bash
# Install test dependencies
uv sync --group test

# Run tests
uv run pytest -q

# Run with verbose output
uv run pytest -vv
```

- Framework: pytest with pytest-asyncio and pytest-xdist (parallel execution)
- Minimum coverage: **80%**
- Test location: `tests/`
- Coverage reports: `./.github/reports/` (XML, JUnit) and `./.github/coverage_html_report/` (HTML)

### Existing Test Coverage

- **VideoDownloader**: parametrized integration tests with URLs from X, Facebook, TikTok
- **ThreadsDownloader**: parametrized integration tests with 9 different Threads.net URLs (including a reply with a multi-level parent chain), plus offline unit tests for the reply-chain extraction logic
- **Economy DB** (`tests/test_economy.py`): per-test isolated SQLite via monkeypatched `_engine`; covers add / settle clamping / transfer atomicity / leaderboard ordering / cooldown rejection edges
- **Blackjack rules** (`tests/test_blackjack.py`): hand-value math (aces, face cards, double-ace demotion), natural Blackjack pays 1.5x, double-Blackjack push, player-bust, dealer-bust, dealer keeps drawing below 17
- **Dice** (`tests/test_dice.py`): seeded RNG determinism, face range, outcome ↔ totals invariant
- **Reward formulas** (`tests/test_video_reward.py`, `tests/test_threads_reward.py`): video reward `min(10 + round(MB), 100)`, success-text DB-failure suffix gating, Threads (user, URL) cooldown claim / expiry / cross-user independence
- **Streaming footer** (`tests/test_gen_reply.py`): regression test for `_handle_streaming` building the `+N 點數` reward suffix and tolerating LiteLLM `output_tokens_details=null`

## CI/CD

| Workflow                    | Trigger            | What It Does                                              |
| --------------------------- | ------------------ | --------------------------------------------------------- |
| `test.yml`                  | Push to main, PRs  | Pytest on Python 3.12 & 3.13, coverage comments on PRs    |
| `code-quality-check.yml`    | PRs                | Pre-commit hooks (Ruff, mypy, etc.)                       |
| `build_image.yml`           | Push to main, tags | Build & push Docker image to `ghcr.io/mai0313/discordbot` |
| `deploy.yml`                | Push to main, tags | Build docs with zensical and deploy to GitHub Pages       |
| `build_release.yml`         | Tags               | Cross-platform binaries via PyInstaller, publish to PyPI  |
| `code_scan.yml`             | Push/PRs           | GitLeaks, Trufflehog, CodeQL security scans               |
| `auto_review_merge.yml`     | PRs                | Auto-review and merge eligible pull requests              |
| `semantic-pull-request.yml` | PRs                | Enforce semantic commit format in PR titles               |
| `auto_labeler.yml`          | PRs                | Auto-label PRs based on changed files                     |
| `release_drafter.yml`       | Push to main       | Auto-draft release notes from merged PRs                  |
| `pre-commit-updater.yml`    | Scheduled          | Auto-update pre-commit hook versions                      |

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`uv run pytest -q`) and linting (`uv run pre-commit run -a`)
5. Commit your changes (`git commit -m 'Add your-feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

## Scripts

| Script                                   | Description                                                           |
| ---------------------------------------- | --------------------------------------------------------------------- |
| `uv run discordbot`                      | Run the bot                                                           |
| `uv run python scripts/artale_data.py`   | Update MapleStory Artale data from `artalemaplestory.com`             |
| `uv run python scripts/prompt_dev.py`    | Iterate prompts against OpenAI / Gemini / Anthropic SDKs              |
| `uv run python scripts/gpt.py`           | Azure GPT-5.4 sandbox — compare chat.completions vs responses APIs    |
| `uv run python scripts/route_dev.py`     | Route-classifier sandbox using `responses.parse` + Pydantic           |
| `uv run python scripts/test_fallback.py` | Smoke-test Litellm `mock_testing_fallbacks` behavior                  |
| `uv run python scripts/video_dev.py`     | Ad-hoc `yt-dlp` download experiments                                  |
| `uv run poe docs`                        | Generate reference docs then serve locally (port 9987)                |
| `make help`                              | Show all available make targets                                       |
| `make clean`                             | Remove build artifacts, caches, reports, and prune repo               |
| `make fmt`                               | Run pre-commit formatting hooks                                       |
| `make test`                              | Run all tests                                                         |
| `make gen-docs`                          | Generate API documentation into `docs/` (mirrors README into `docs/`) |
| `make uv-install`                        | Install uv package manager on the system                              |
| `make submodule-init`                    | Initialize and update git submodules                                  |
| `make submodule-update`                  | Update all submodules to latest remote version                        |

## License

[MIT](LICENSE)
