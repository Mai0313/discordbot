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
│   ├── gen_reply.py         # AI chat — @mention/DM trigger, routing, streaming via OpenAI Responses API
│   ├── _gen_reply/
│   │   └── prompts.py       # REPLY / ROUTE / SUMMARY / IMAGE / HISTORY prompts
│   ├── help.py              # /help slash command (localized guide)
│   ├── log_msg.py           # Message logging to SQLite (PostgreSQL ready)
│   ├── maplestory.py        # /maple_* slash commands (8 commands)
│   ├── _maplestory/
│   │   ├── constants.py     # Display templates for stats
│   │   ├── embeds.py        # Discord embed builders
│   │   ├── models.py        # Pydantic data models
│   │   ├── service.py       # Data loading, search logic, caching
│   │   └── views.py         # Interactive UI components (dropdown select)
│   ├── parse_threads.py     # Threads.net auto-parser
│   ├── template.py          # /ping and utility reactions
│   └── video.py             # /download_video slash command
├── typings/                 # Pydantic configuration models
│   ├── config.py            # DiscordConfig
│   ├── database.py          # SQLite / PostgreSQL / Redis configs
│   └── llm.py               # LLM endpoint config (BASE_URL / API_KEY)
└── utils/
    ├── downloader.py        # yt-dlp video downloader wrapper
    └── threads.py           # Threads.net content scraper

scripts/
├── artale_data.py           # Scrape Artale MapleStory data from artalemaplestory.com
├── gen_docs.py              # Generate mkdocstrings reference pages
├── gpt.py                   # Azure GPT-5.4 sandbox comparing chat.completions vs responses API
├── migrate.py               # Database migration helper (SQLite → PostgreSQL)
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
└── threads/                 # Downloaded Threads.net media
```

### Architecture

- **Cog-based**: Each feature is a separate cog in `cogs/`. The bot auto-discovers and loads all `.py` files in the directory (excluding `__` prefixed files). Helper packages live in sibling `_<cog>/` folders so they are not auto-loaded.
- **Async**: Built on nextcord with async/await patterns throughout.
- **Config**: Pydantic models + `pydantic-settings` load from `.env` automatically (`DiscordConfig`, `LLMConfig`, `DatabaseConfig`).
- **Logging**: `setup_logging()` in `discordbot/__init__.py` configures `logfire` (local console only, `send_to_logfire=False`) and tees stdout to `./data/logs/<timestamp>.log` for each run. `nextcord.state` logs are forwarded into logfire too.
- **LLM client**: A single `AsyncOpenAI` client (`base_url=BASE_URL`, `api_key=API_KEY`) issues all chat / image / video calls. The endpoint is OpenAI-compatible — typically a Litellm proxy that fronts Gemini / Claude / OpenAI / etc.
- **AI Routing**: The `gen_reply` cog uses a fast model to classify user intent (QA, IMAGE, VIDEO, SUMMARY) via `client.responses.create` and dispatches to the matching handler. All chat / route / caption calls use the **OpenAI Responses API** (not Chat Completions); the slow reply path enables model-specific tools (Gemini → `googleSearch` + `urlContext`; Claude → `web_search_*` + `web_fetch_*`; others → OpenAI `web_search`) and streams the answer event-by-event (`response.output_text.delta`). Each response ends with a Discord-quoted footer (`> **{model}** ⬆ in ⬇ out $cost`) where the cost comes from `litellm.model_cost`. Processing progress is shown via emoji reactions on the user's message (🤔 → 🔀 → 🎨/🎬/📖/❓ → 🆗, or ❌ on error).
- **Trigger rule**: In DMs the bot always responds; in guilds it only responds when the message text contains `<@bot_id>` (a reply-notification alone is ignored, so users replying to a Threads embed or a download result won't accidentally summon the bot).
- **Attachment ingestion**: `_get_attachments` collects images from message attachments, stickers, and Discord embeds (preferring `media.discordapp.net` proxy URLs, since CDNs like Threads expire and reject unauthenticated requests), so the AI can see images inside referenced/quoted bot embeds (e.g. a Threads-parsed post). Video attachments are currently skipped to avoid uploading large blobs to the LLM.

## Code Standards

### Tooling

| Tool           | Purpose                                   |
| -------------- | ----------------------------------------- |
| **Ruff**       | Linting and formatting (line length: 99)  |
| **mypy**       | Type checking with Pydantic plugin        |
| **ty**         | Astral's type checker (error-level rules) |
| **pre-commit** | Runs all checks automatically on commit   |

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

Hooks include: Ruff, mypy, ShellCheck, mdformat, codespell, gitleaks, nbstripout, uv-sync, uv-lock.

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
- **ThreadsDownloader**: parametrized integration tests with 7 different Threads.net URLs

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
| `uv run python scripts/migrate.py`       | Migrate logged messages from SQLite to PostgreSQL                     |
| `uv run python scripts/prompt_dev.py`    | Iterate prompts against OpenAI / Gemini / Anthropic SDKs              |
| `uv run python scripts/gpt.py`           | Azure GPT-5.4 sandbox — compare chat.completions vs responses APIs    |
| `uv run python scripts/route_dev.py`     | Route-classifier sandbox using `responses.parse` + Pydantic           |
| `uv run python scripts/test_fallback.py` | Smoke-test Litellm `mock_testing_fallbacks` behavior                  |
| `uv run python scripts/video_dev.py`     | Ad-hoc `yt-dlp` download experiments                                  |
| `uv run poe docs`                        | Generate reference docs then serve locally (port 9987)                |
| `make help`                              | Show all available make targets                                       |
| `make clean`                             | Remove build artifacts, caches, reports, and prune repo               |
| `make format`                            | Run pre-commit formatting hooks                                       |
| `make test`                              | Run all tests                                                         |
| `make gen-docs`                          | Generate API documentation into `docs/` (mirrors README into `docs/`) |
| `make uv-install`                        | Install uv package manager on the system                              |
| `make submodule-init`                    | Initialize and update git submodules                                  |
| `make submodule-update`                  | Update all submodules to latest remote version                        |

## License

[MIT](LICENSE)
