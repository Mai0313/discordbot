# Project Guidelines

## Build and Test

- Use `uv` for Python environment and dependency management.
- Install dependencies with `uv sync`.
- Run the bot with `uv run discordbot`.
- Run tests with `uv sync --group test` and `uv run pytest -q`.
- Keep test coverage at or above the configured 80% threshold in `pyproject.toml`.
- Run formatting and repository checks with `make format` before finishing changes.
- Generate docs with `make gen-docs` when changing public modules or scripts that affect documentation.

## Architecture

- The main entry point is `src/discordbot/cli.py`, which defines `DiscordBot` and dynamically loads cogs from `src/discordbot/cogs`.
- Keep Discord features inside cog modules under `src/discordbot/cogs/`.
- Keep reusable integrations and helpers under `src/discordbot/utils/`.
- Keep configuration models under `src/discordbot/typings/` using Pydantic settings classes loaded from `.env`.
- Keep repository scripts in `scripts/`; `scripts/artale_data.py` depends on Playwright and is separate from runtime bot behavior.

## Conventions

- Follow the existing async-first nextcord style: slash commands live in cog classes and handlers are `async def`.
- When adding slash commands, include localized names and descriptions when the surrounding code does.
- Prefer `logfire` for application logging and observability instead of introducing new ad-hoc logging patterns.
- Reuse the existing Pydantic settings approach for configuration; do not hardcode tokens, API keys, or service URLs.
- Keep bot data and generated artifacts under `data/` rather than scattering files elsewhere.
- Match the current Ruff-driven style in `pyproject.toml`: 99-character lines, double quotes, Google-style docstrings, and type annotations where practical.
- Preserve the existing testing style in `tests/`: small pytest tests with mocks or dry-run paths instead of real network-heavy execution.

## Environment Notes

- Required local secrets live in `.env`; use `.env.example` as the template.
- The bot requires `DISCORD_BOT_TOKEN` and an OpenAI-compatible API configuration to run.
- Video download features rely on `ffmpeg`, and MapleStory data updates require `uv run playwright install chromium` before running the scraper.
- Optional database services are configured through `POSTGRES_URL`, `REDIS_URL`, and `SQLITE_FILE_PATH`.
- Be careful with downloader changes: the project already includes site-specific handling such as Facebook URL expansion and low-quality fallback for Discord file-size limits.
