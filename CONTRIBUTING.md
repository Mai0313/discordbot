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

# Install all dependencies (including dev and test groups)
uv sync --dev

# Set up environment
cp .env.example .env
# Edit .env with your tokens and API keys

# Run the bot
uv run discordbot
```

### Optional: MapleStory Data

```bash
uv run update
```

## Project Structure

```
src/discordbot/
├── cli.py                  # Main bot entry point (DiscordBot class)
├── cogs/                   # Command modules (auto-loaded)
│   ├── gen_reply.py        # AI chat — @mention trigger, streaming, routing
│   │   └── _gen_reply/
│   │       └── prompts.py  # System & routing prompts
│   ├── parse_threads.py    # Threads.net auto-parser
│   ├── video.py            # /download_video slash command
│   ├── maplestory.py       # /maple_* slash commands
│   │   └── _maplestory/
│   │       ├── service.py  # Search logic
│   │       ├── models.py   # Data models
│   │       ├── embeds.py   # Discord embed builders
│   │       ├── constants.py
│   │       └── views.py    # Interactive UI components
│   ├── log_msg.py          # Message logging to SQLite/PostgreSQL
│   └── template.py         # /ping and utility reactions
├── typings/                # Pydantic configuration models
│   ├── config.py           # DiscordConfig
│   ├── database.py         # SQLite/PostgreSQL/Redis configs
│   └── llm.py              # LLM endpoint config
└── utils/
    ├── downloader.py       # yt-dlp video downloader wrapper
    └── threads.py          # Threads.net content scraper
data/
├── monsters.json           # MapleStory monster/item database
└── downloads/              # Temporary video download storage
```

### Architecture

- **Cog-based**: Each feature is a separate cog in `cogs/`. The bot auto-discovers and loads all `.py` files in the directory (excluding `__` prefixed files).
- **Async**: Built on nextcord with async/await patterns throughout.
- **Config**: Pydantic models + `pydantic-settings` load from `.env` automatically.
- **AI Routing**: The `gen_reply` cog uses a fast model to classify user intent (QA, IMAGE, VIDEO, SUMMARY) and routes to the appropriate handler. Processing progress is shown via emoji reactions on the user's message (🤔 → 🔀 → route emoji → 🆗).

## Code Standards

### Tooling

| Tool           | Purpose                                  |
| -------------- | ---------------------------------------- |
| **Ruff**       | Linting and formatting (line length: 99) |
| **mypy**       | Type checking with Pydantic plugin       |
| **pre-commit** | Runs all checks automatically on commit  |

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

Hooks include: Ruff, mypy, ShellCheck, mdformat, codespell, gitleaks, nbstripout.

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

- **TemplateCogs**: message reactions and `/ping` embed
- **ReplyGeneratorCogs**: `_get_attachment_list()` and message processing
- **VideoCogs**: `/download_video` happy path (downloader mocked)

## CI/CD

| Workflow                 | Trigger            | What It Does                                              |
| ------------------------ | ------------------ | --------------------------------------------------------- |
| `test.yml`               | Push to main, PRs  | Pytest on Python 3.12 & 3.13, coverage comments on PRs    |
| `code-quality-check.yml` | PRs                | Pre-commit hooks (Ruff, mypy, etc.)                       |
| `build_image.yml`        | Push to main, tags | Build & push Docker image to `ghcr.io/mai0313/discordbot` |
| `deploy.yml`             | Push to main, tags | Build mkdocs and deploy to GitHub Pages                   |
| `build_release.yml`      | Tags               | Cross-platform binaries via PyInstaller, publish to PyPI  |
| `code_scan.yml`          | Push/PRs           | GitLeaks, Trufflehog, CodeQL, Trivy security scans        |

## How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`uv run pytest -q`) and linting (`uv run pre-commit run -a`)
5. Commit your changes (`git commit -m 'Add your-feature'`)
6. Push to the branch (`git push origin feature/your-feature`)
7. Open a Pull Request

## Scripts

| Script                | Description                               |
| --------------------- | ----------------------------------------- |
| `uv run discordbot`   | Run the bot                               |
| `uv run update`       | Install Chromium + update MapleStory data |
| `make format`         | Run pre-commit formatting hooks           |
| `make test`           | Run all tests                             |
| `make gen-docs`       | Generate API documentation                |
| `uv run mkdocs serve` | Serve documentation locally (port 9987)   |

## License

[MIT](LICENSE)
