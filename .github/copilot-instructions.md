# Copilot Instructions

## Architecture Overview

This is a **nextcord**-based Discord bot. The entry point is `src/discordbot/cli.py` (`DiscordBot` class extending `commands.Bot`). On startup, all non-`__`-prefixed `.py` files in `src/discordbot/cogs/` are auto-loaded as extensions.

```
src/discordbot/
  cli.py                  # DiscordBot class, cog auto-loader, event handlers
  cogs/                   # One file = one cog; complex features use cogs/_feature/ subpackage
    gen_reply.py          # AI chat (mention-triggered streaming replies)
    video.py              # yt-dlp video downloader
    maplestory.py         # MapleStory slash commands (delegates to _maplestory/)
    _maplestory/          # Service/model/embed/view layer for MapleStory feature
  typings/
    config.py             # DiscordConfig (Pydantic BaseSettings)
    database.py           # PostgreSQLConfig, RedisConfig, SQLiteConfig
  utils/
    llm.py                # LLMSDK – wraps AsyncOpenAI / AsyncAzureOpenAI
    downloader.py         # VideoDownloader (yt-dlp)
```

## Developer Workflows

```bash
uv sync           # Install/update all dependencies
make test         # Run pytest with coverage (requires ≥80%)
make format       # Run pre-commit hooks (ruff check + format, yaml/json linters)
```

Tests live in `tests/`, use `asyncio_mode = "auto"` (no explicit `@pytest.mark.asyncio` needed).

## Adding a New Cog

1. Create `src/discordbot/cogs/my_feature.py`.
2. Define a class inheriting `commands.Cog`, implement `__init__(self, bot)`.
3. Add a module-level `async def setup(bot): bot.add_cog(MyFeatureCogs(bot), override=True)`.
4. The bot auto-discovers it on next start — no registration needed.

For complex features, create a `cogs/_my_feature/` subpackage (models, service, embeds, views) and import from the top-level cog file, following the pattern in `cogs/_maplestory/`.

## Slash Command Conventions

- Always call `await interaction.response.defer()` first for any non-trivial handler.
- Include `name_localizations` and `description_localizations` (at minimum `Locale.zh_TW` and `Locale.ja`).
- Use `await interaction.edit_original_message(...)` to update progress/results.

```python
@nextcord.slash_command(
    name="my_cmd",
    name_localizations={Locale.zh_TW: "我的指令", Locale.ja: "マイコマンド"},
    ...
)
async def my_cmd(self, interaction: Interaction) -> None:
    await interaction.response.defer()
    ...
    await interaction.followup.send(embed=embed)
```

## Configuration & Environment

All config classes extend Pydantic `BaseSettings` with `AliasChoices` for env var aliasing. Config is instantiated directly (no DI container); `dotenv.load_dotenv()` is called at module top.

Copy `.env.example` to `.env`. Required vars:

| Variable                                    | Purpose      |
| ------------------------------------------- | ------------ |
| `DISCORD_BOT_TOKEN`                         | Bot token    |
| `OPENAI_BASE_URL` / `AZURE_OPENAI_ENDPOINT` | LLM endpoint |
| `OPENAI_API_KEY` / `AZURE_OPENAI_API_KEY`   | LLM key      |

Azure OpenAI is auto-detected when `OPENAI_API_VERSION` is set; set `DISCORD_TEST_SERVER_ID` to speed up slash-command sync in dev.

## LLM Integration

`LLMSDK` in `utils/llm.py` is a `BaseSettings` subclass; instantiate with a model name and the client is auto-constructed via `@computed_field @cached_property`. The default model in `gen_reply.py` is `gemini-3-pro-preview`. For streaming replies, poll `AsyncStream[ChatCompletionChunk]` and edit the Discord message every ~10 characters.

## Logging

Use `logfire` everywhere — **not** `print` or standard `logging.info`. The nextcord logger bridges into logfire via `LogfireLoggingHandler`.

```python
logfire.info("Event occurred", key=value)
logfire.error("Something failed", _exc_info=True)
```

## Code Quality

Ruff is the sole linter/formatter (configured in `pyproject.toml`). Run via `make format` or pre-commit. CI enforces both `ruff check` and `ruff format`.
