# Contributing

Thanks for taking the time to improve this project. This guide focuses on the
commands and conventions needed to send a pull request.

## Local setup

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

Useful local checks:

```bash
uv run pytest
uv run pre-commit run -a
make fmt
make gen-docs
```

`make fmt` is the same project-level check as `uv run pre-commit run -a`.
`make gen-docs` regenerates `docs/` from the README files, `CONTRIBUTING.md`,
and the Python sources.

## Project layout

- `src/discordbot/`: bot runtime, cogs, shared typings, and utilities.
- `src/discordbot/cogs/`: nextcord cogs. Helper packages use sibling
    `_<cog>/` directories so they are not auto-loaded.
- `tests/`: pytest suite.
- `scripts/`: data, prompt, route, docs, and downloader development helpers.
- `data/`: local runtime data such as SQLite databases, logs, model price cache,
    and temporary media files. Do not commit generated runtime data.
- `docker/`: container build files.
- `.github/workflows/`: CI, release, security scan, and automation workflows.

## Development workflow

1. Fork the repository and create a focused branch, for example
    `feature/your-change`, `fix/your-bug`, `docs/your-doc-change`, or
    `chore/your-maintenance-task`.
2. Keep the change scoped. Avoid unrelated refactors in the same pull request.
3. Add or update tests when behavior changes.
4. Update user-facing docs when commands, configuration, or visible behavior
    changes. For slash-command behavior, update the `/help` content in
    `src/discordbot/cogs/help.py` in the same pull request.
5. Run the local checks before opening the pull request:

```bash
uv run pytest
uv run pre-commit run -a
```

Pull request titles and commit messages should follow
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/), for
example:

```text
feat: add blackjack surrender option
fix(economy): prevent duplicate blackjack settlement
docs: simplify contributor guide
```

## Code conventions

- Use existing project patterns before adding new abstractions.
- Keep cog `setup(bot)` functions synchronous. The bot loads cogs before the
    first application-command sync.
- Use the OpenAI Responses API through the existing `AsyncOpenAI` clients for
    LLM request paths.
- Keep model names behind the existing `ModelSettings` properties instead of
    hardcoding them at call sites.
- Use Pydantic models for structured data.
- Prefer precise typed APIs over `Any`.
- Use `logfire.info`, `logfire.warn`, or `logfire.error(..., _exc_info=True)`
    for new logging.
- Do not commit secrets, `.env`, generated databases, logs, downloads, or other
    runtime artifacts from `data/`.

## Tests and quality gates

The test suite is configured in `pyproject.toml` and runs with pytest,
pytest-asyncio, pytest-xdist, and coverage.

```bash
uv run pytest
```

Coverage must stay at or above 80%. CI runs tests on Python 3.12 and 3.13 for
pull requests and pushes to protected branches, except for documentation,
chore, and CI-only branch prefixes that are intentionally skipped by the test
workflow.

Pre-commit is the main quality gate:

```bash
uv run pre-commit run -a
```

It runs Ruff formatting and linting, mypy, Markdown formatting, ShellCheck,
codespell, gitleaks, uv lock checks, and standard file hygiene hooks.

## Releases

Maintainers handle releases through GitHub Actions.

- Merged changes on `main` update the draft release notes.
- Tags matching `v*` build release artifacts and publish the Docker image.
- The release workflow builds cross-platform binaries and publishes the Python
    package when credentials are available.

Contributors usually do not need to run release commands locally.

## License

By contributing, you agree that your contribution is licensed under the
[MIT License](LICENSE).
