# Contributing

Thanks for improving this project. This guide covers the local setup, workflow, and conventions expected for pull requests.

## Local Setup

Prerequisites:

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` for video download and merge features

Set up the repository:

```bash
git clone https://github.com/Mai0313/discordbot.git
cd discordbot
uv sync --all-groups
cp .env.example .env
```

Fill in the Discord and OpenAI-compatible endpoint values in `.env`.

Run the bot:

```bash
uv run discordbot
```

Useful checks:

```bash
uv run pytest
uv run pre-commit run -a
make fmt
make gen-docs
```

`make fmt` runs the same project-level check as `uv run pre-commit run -a`. `make gen-docs` regenerates `docs/` from the README files, `CONTRIBUTING.md`, and Python sources.

## Project Layout

- `src/discordbot/cli.py`: bot entry point, intent setup, cog loading, global message reward, and application-command sync.
- `src/discordbot/cogs/`: nextcord cogs. Sibling `_<cog>/` packages hold cog-private helpers and are not auto-loaded.
- `src/discordbot/typings/`: shared Pydantic models, settings, enums, and pure domain types.
- `src/discordbot/utils/`: downloader, image, Threads, and LiteLLM pricing helpers.
- `tests/`: pytest suite.
- `scripts/`: local maintenance and development tools.
- `data/`: runtime SQLite databases, logs, cached prices, downloads, and scratch files. Do not commit generated runtime data.
- `docker/` and `docker-compose.yaml`: container build and runtime setup.
- `.github/workflows/`: CI, code quality, docs deploy, release, and image publishing workflows.

## Workflow

- Create a focused branch such as `feature/your-change`, `fix/your-bug`, `docs/your-doc-change`, or `chore/your-maintenance-task`.
- Keep PRs scoped. Avoid unrelated refactors.
- Use Conventional Commits for commit messages and PR titles:

```text
feat: add blackjack surrender option
fix(economy): prevent duplicate settlement
docs: simplify user README
```

- Add or update tests for behavior changes.
- Update user-facing docs when commands, configuration, or visible behavior changes.
- For slash-command behavior, update `_HELP_CONTENT` in `src/discordbot/cogs/help.py` in the same change and keep `tests/test_help.py` passing.
- Run local checks before opening the PR:

```bash
uv run pytest
uv run pre-commit run -a
```

## Code Conventions

- Follow existing project patterns before adding a new abstraction.
- Use `logfire.info`, `logfire.warn`, or `logfire.error(..., _exc_info=True)` for new logging.
- Keep cog `setup(bot)` functions synchronous:

```python
def setup(bot: commands.Bot) -> None:
    bot.add_cog(MyCog(bot), override=True)
```

`async def setup` is not safe here because nextcord schedules it without awaiting it, which can leave cogs unregistered before the first slash-command sync.

- Slash commands need `name_localizations` and `description_localizations` for `en-US`, `zh-TW`, and `ja` where applicable.
- Do not import peer cogs directly. Use the bot instance, shared typings, or helper modules under the cog's private `_<cog>/` package.
- Use Pydantic for structured data models. Prefer `BaseModel`, frozen models, enums, and typed result objects over dictionaries or `dataclass`.
- Environment-backed settings should use `pydantic_settings.BaseSettings` with explicit `validation_alias=AliasChoices("ENV_NAME")`.
- Keep `Field(description=..., examples=...)` populated for configurable values. These descriptions document the environment contract.
- Prefer precise typed APIs. `Any` is a last resort.
- Keyword arguments are required for normal function calls, including single argument calls such as `create_engine(url=...)` and `re.compile(pattern=...)`.
- Do not add a bare `*` to new function signatures only to force keyword-only calls unless an external API or correctness issue needs it.
- Accept normal positional-only idioms such as `len(value)`, `str(value)`, `Path("file")`, exception constructors, variadic collectors, and `logfire.info("message")`.
- Avoid intermediate one-level aliases when directly using the original object is clearer.
- Do not blanket `# noqa`. Use the narrowest rule-specific ignore with a short reason.
- Keep comments focused on non-obvious behavior. Do not narrate the code or reference PR numbers in comments.

## LLM And Media Paths

- Runtime LLM calls go through `AsyncOpenAI` clients and the OpenAI Responses API.
- `OPENAI_BASE_URL` usually points at LiteLLM. Provider-specific behavior should be expressed through model names, `ModelSettings`, tools, or `extra_body`.
- Do not import provider-native SDKs such as `google-genai` or `anthropic` into request paths. Development scripts may use them.
- Runtime model strings for `./src` live in `RuntimeModelCatalog` in `src/discordbot/typings/models.py`; update that catalog instead of hardcoding names at call sites.
- Preserve the reaction-based progress UX for AI replies and parsers. The bot should not send intermediate "thinking" messages.
- Video delivery keeps progress text on the deferred original message, then edits that same message with the final file and source URL.

## Economy And Games

- `data/economy.db`, `data/global_state.db`, and `data/messages.db` are separate SQLite databases. Keep `economy.db` tables user-scoped with `user_id` and `name`; put bot-wide shared state such as jackpot pools in `global_state.db`.
- Economy helpers use a module-level SQLAlchemy engine so tests can monkeypatch the engine object.
- 虛擬歡樂豆 balances are cross-server. Do not add `guild_id` to the account model.
- `UserAccount.avatar_url` is a last-seen cache. Discord-facing write paths should pass `guild_avatar_url(...)` with guild context so guild avatars are stored when available, then fall back to the global `display_avatar`. Existing rows are not backfilled; they refresh naturally on later writes.
- `credit_with_repayment` is the income path for message reward, chat reward, and casino payout. Long-term loans are repaid explicitly through loan helpers; passive income and gifts do not auto-repay debt.
- Long-term loans live in `loan_proposal` and `loan_contract`. Personal credit requests are borrower-initiated and debit the lender on acceptance, and central-bank loans mint borrower balance through central-banker button approval.
- Central banker access is stored on `UserAccount.is_central_banker` and managed out-of-band with `scripts/manage_central_banker.py`, separate from Discord-side economy admins.
- Casino settlement is atomic. Validate or clamp bets before play, then apply the signed result once through the settlement helpers. Player-side casino losses clamp at balance 0; dealer / house ledgers may still go negative.
- Jackpot settlements coordinate writes across `economy.db` and `global_state.db`; ordinary exceptions roll both sessions back before either commit, but SQLite cannot make a hard crash between two database-file commits cross-file atomic.
- Daily casino loss leaderboards read persisted `casino_account` counters. Keep those counters tied to player-side casino settlement deltas only.
- `UserAccount.hide_from_leaderboard` defaults to `False`. Public balance and daily loss leaderboards omit rows where it is set; maintenance code should opt into hidden rows when it needs a true full-account sweep.
- Blackjack house ledger and Dragon Gate jackpot pool are separate counterparties. Do not route Dragon Gate through the house ledger.
- Interactive game, public economy, and public stock responses are tracked for restart cleanup and expire after settlement or timeout. Private balance, loan, check-in, VIP, and admin-error replies are not tracked.

## Tests And Quality Gates

The pytest configuration lives in `pyproject.toml`.

```bash
uv run pytest
```

Coverage must stay at or above 80%. CI runs tests on Python 3.12 and 3.13 for pushes and pull requests targeting `main`, `master`, or `release/*`, except for documentation, chore, and CI branch prefixes intentionally skipped by the test workflow.

The pre-commit gate is the canonical local quality check:

```bash
uv run pre-commit run -a
```

It runs Ruff formatting and linting, mypy with the Pydantic plugin, Markdown formatting, ShellCheck, codespell, gitleaks, uv lock checks, and standard file hygiene hooks.

## Documentation

- `README.md` is the canonical user-facing README.
- `README.zh-CN.md` and `README.zh-TW.md` should mirror the English README structure.
- `CONTRIBUTING.md` is developer-facing and stays in English.
- `CLAUDE.md` is AI-agent-facing. Keep it dense and project-specific.
- `docs/` is generated by `make gen-docs`. Do not hand-edit generated docs.

## Text Formatting

- Do not reflow human-written prose.
- Do not hard-wrap Markdown or documentation text to 72, 80, or 100 columns. Editors should handle visual wrapping.
- When modifying documents, make the smallest textual diff possible and preserve the surrounding line structure.
- A prose paragraph should usually stay on one logical line unless the existing file is intentionally and consistently manual-wrapped.

## Releases

Maintainers handle releases through GitHub Actions.

- Merged changes on `main` update draft release notes.
- Tags matching `v*` build release artifacts and publish the Docker image.
- The release workflow builds cross-platform binaries and publishes the Python package when credentials are available.

Contributors usually do not need to run release commands locally.

## License

By contributing, you agree that your contribution is licensed under the [MIT License](LICENSE).
