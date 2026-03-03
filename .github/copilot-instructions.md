# GitHub Copilot Instructions

## Project Overview

AI-powered Discord bot built with **nextcord**, using a modular cog-based architecture. Key layers: Discord event handling → Cog commands → LLM/service logic → external APIs (OpenAI/Azure, yt-dlp, PostgreSQL/Redis/SQLite).

## Architecture

### Core Entry Point
`src/discordbot/cli.py` — `DiscordBot` extends `nextcord.ext.commands.Bot`. Cogs are **auto-discovered and loaded** from `src/discordbot/cogs/*.py` (any file not prefixed with `__`). No manual registration needed.

### Adding a New Cog
1. Create `src/discordbot/cogs/my_feature.py` with a class extending `commands.Cog`
2. Add a module-level `setup(bot)` function:
   ```python
   def setup(bot: commands.Bot) -> None:
       bot.add_cog(MyFeatureCogs(bot))
   ```
3. For complex cogs, use a private subpackage (e.g., `cogs/_maplestory/`) with `service.py`, `models.py`, `embeds.py`, `views.py`.

### Configuration Pattern
All config classes inherit from **`pydantic_settings.BaseSettings`** and use `AliasChoices` for environment variable mapping. See `src/discordbot/typings/config.py` and `src/discordbot/utils/llm.py`:
```python
api_key: str = Field(..., validation_alias=AliasChoices("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"))
```
Use `dotenv.load_dotenv()` at the top of each settings file.

### LLM Integration
`src/discordbot/utils/llm.py` — `LLMSDK` (a `BaseSettings`) auto-selects `AsyncOpenAI` vs `AsyncAzureOpenAI` based on whether `OPENAI_API_VERSION` is set. Default model in `gen_reply.py` is `gemini-3-pro-preview`. Stream responses are default; update Discord messages approximately every 10 characters.

## Slash Command Conventions

- Always call `await interaction.response.defer()` before any async work to prevent timeout
- Use `name_localizations` and `description_localizations` with `Locale.zh_TW` and `Locale.ja`
- Example pattern from `cogs/video.py`:
  ```python
  @nextcord.slash_command(name="cmd", name_localizations={Locale.zh_TW: "中文名"}, dm_permission=True)
  async def cmd(self, interaction: Interaction, ...) -> None:
      await interaction.response.defer()
      await interaction.followup.send("...")
      await interaction.edit_original_message(content="done", file=...)
  ```

## Logging

Use **`logfire`** (not Python's standard `logging`) for structured logging throughout:
```python
logfire.info("Event", key=value)
logfire.warn("Warning message")
logfire.error("Error", _exc_info=True)
```

## Developer Workflows

```bash
uv sync                # Install dependencies
make test              # Run pytest with coverage (requires ≥80%)
make format            # Run pre-commit hooks (ruff lint/format)
make gen-docs          # Generate API docs from src/ and scripts/
```

Tests use `asyncio_mode = "auto"` — async test functions work without decorators.

## Environment Variables

Copy `.env.example` to `.env`. Key variables:
- `DISCORD_BOT_TOKEN`, `DISCORD_TEST_SERVER_ID`
- `OPENAI_BASE_URL` / `OPENAI_API_KEY` (or Azure equivalents)
- `POSTGRES_URL`, `REDIS_URL`, `SQLITE_FILE_PATH`
- `YOUTUBE_DATA_API_KEY`

The bot creates `./data/` on startup. MapleStory data must be at `./data/monsters.json`.

## Key Dependencies

| Package | Purpose |
|---|---|
| `nextcord` | Discord bot framework |
| `openai` / `ag2` (autogen) | LLM calls + image utilities (`get_pil_image`, `pil_to_data_uri`) |
| `yt-dlp` | Video downloading in `utils/downloader.py` |
| `pydantic-settings` | All config/settings classes |
| `logfire` | Structured logging/observability |
| `sqlalchemy` / `redis` | Database backends |
