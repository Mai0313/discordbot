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

Fill in the Discord and OpenAI-compatible endpoint values in `.env`. Set `GEMINI_API_KEY` (a Google AI Studio key) to enable the direct-to-Google features: video and music generation, Gemini Files API uploads of attachments and linked-post media, YouTube video answers, and deep research.

Run the bot:

```bash
uv run discordbot
```

Useful checks:

```bash
uv run pytest
uvx pre-commit run -a
make fmt
make gen-docs
```

`make fmt` runs the same project-level check as `uvx pre-commit run -a`. `make gen-docs` regenerates `docs/` from the README files, `CONTRIBUTING.md`, and Python sources.

## Project Layout

- `src/discordbot/cli.py`: bot entry point, intent setup, cog loading, global message reward, and application-command sync.
- `src/discordbot/cogs/`: nextcord cogs, one directory each. `cogs/<name>/cog.py` is the module the loader imports; everything beside it in that directory is that cog's own code.
- `src/discordbot/services/`: domain engines shared by more than one cog (the economy ledger, the memory store, the per-platform link readers in `services/platforms/`). Discord-free, and never imports from `cogs/`, both enforced by `tests/test_package_layering.py`.
- `src/discordbot/typings/`: shared Pydantic models, settings, enums, and pure domain types.
- `src/discordbot/utils/`: generic helpers with no domain state — images, embeds, LiteLLM pricing, the scratch directory, the link-error vocabulary.
- `tests/`: pytest suite.
- `scripts/`: local maintenance and development tools. The `*_dev.py` scripts are smoke tests against the live APIs, using the credentials in `.env` the way the bot does, so every run is a paid call; edit the call under `if __name__ == "__main__":` and run one with `uv run python -m scripts.<name>`.
- `data/`: runtime data; SQLite databases live in `data/database/`, alongside logs, cached prices, and other runtime files. Do not commit generated runtime data.
- `docker/` and `docker-compose.yaml`: container build and runtime setup.
- `.github/workflows/`: CI, code quality, docs deploy, release, and image publishing workflows.

## Workflow

- Create a focused branch such as `feat/your-change`, `fix/your-bug`, `docs/your-doc-change`, or `chore/your-maintenance-task`.
- Keep PRs scoped. Avoid unrelated refactors.
- Use Conventional Commits for commit messages and PR titles:

```text
feat: add blackjack surrender option
fix(economy): prevent duplicate settlement
docs: simplify user README
```

- Add or update tests for behavior changes.
- Update user-facing docs when commands, configuration, or visible behavior changes.
- For slash-command behavior, update `src/discordbot/cogs/gen_reply/capabilities.md` and the command table of all three READMEs in the same change, and keep `tests/test_capabilities.py` passing. That English document is injected into the reply so the bot can answer "what can you do" in the asker's language; the guard covers group subcommands, so each `/<group> <subcommand>` needs its own line there.
- Run local checks before opening the PR:

```bash
uv run pytest
uvx pre-commit run -a
```

## Code Conventions

- Follow existing project patterns before adding a new abstraction.
- Pick a log level with the ladder in [Logging](#logging) below.
- Keep cog `setup(bot)` functions synchronous:

```python
# src/discordbot/cogs/<name>/cog.py
def setup(bot: commands.Bot) -> None:
    bot.add_cog(MyCog(bot), override=True)
```

`async def setup` is not safe here: cogs load inside `DiscordBot()` before any event loop runs, so nextcord's attempt to schedule it raises and boot aborts with `ExtensionFailed`.

- Slash commands take their English `name` and `description` as the defaults, plus `name_localizations` and `description_localizations` for `Locale.zh_TW` and `Locale.ja`.
- A cog directory holds one cog's code. Do not import anything from a peer cog's directory: use the bot instance, `typings/`, `utils/`, or promote the shared part into `services/`. `tests/test_package_layering.py` enforces this.
- Use Pydantic for structured data models. Prefer `BaseModel`, frozen models, enums, and typed result objects over dictionaries or `dataclass`.
- Environment-backed settings should use `pydantic_settings.BaseSettings` with explicit `validation_alias=AliasChoices("ENV_NAME")`.
- Keep `Field(description=..., examples=...)` populated for configurable values. These descriptions document the environment contract.
- Prefer precise typed APIs. `Any` is a last resort.
- Keyword arguments are required for normal function calls, including single argument calls such as `create_engine(url=...)` and `re.compile(pattern=...)`.
- Do not add a bare `*` to new function signatures only to force keyword-only calls unless an external API or correctness issue needs it.
- Accept normal positional-only idioms such as `len(value)`, `str(value)`, `Path("file")`, exception constructors, variadic collectors, and `logfire.info("message")`.
- Avoid intermediate one-level aliases when directly using the original object is clearer.
- Do not blanket `# noqa`. Use the narrowest rule-specific ignore with a short reason.

## Comments

A comment earns its place only when deleting it would let the next reader break the code. Write the trap, the invariant, or the constraint that is not visible from the code. Everything else is what `git log`, the pull request and the issue are for.

**One fact, one home.** The home is the file that changes when the fact changes. Prose about a set of files belongs to the module that owns the set; its members carry a pointer that never needs editing.

Four things therefore never appear in a comment or docstring:

- **A count of anything outside this file.** "one feature five times over", "all five cogs", "three of the six", "exactly eleven kill-switches". Write "every expansion cog" — the registry, the base class or the test is the enumeration, and it cannot go stale.
- **A list of sibling modules, callers or test files.** A caller list is wrong the day someone adds a caller, and nothing checks it.
- **History.** "#636 deleted this", "was a prompt rule, now a code sweep", "this used to". The reason survives; the story of how it got here does not.
- **A measurement's provenance.** Keep the conclusion the number justifies, drop where it came from. A date survives only for a claim about an external system that moves under us — a provider limit, a platform's page shape — where staleness is the point.

**Comparing this file's number to another file's is coupling.** Say what this number is for. The exception is a constant defined as an expression over another, where the relationship *is* the code.

**Dropping a count is not the same as writing a universal.** "Douyin and Bilibili check this, Threads does not" becomes "every source that has a gate checks this" and is now false, because the one source that does not is a source with a gate. Name the exception: an outlier a reader has to know about is a fact, not a roster, and it is the half worth keeping when the list goes.

**None of this deletes the only copy of a reason.** Where the shortest true form of a reason is a piece of history, a measured figure or a name from another file, keep it and compress it. A warning, a "do not", and a "re-measure before changing this" are reasons rather than narration: the comment saying a column cannot be dropped on a deployed database is the only thing stopping someone dropping it.

**A docstring is the contract**: what it does, what it needs, what it returns, what can go wrong. A `BaseModel` whose fields all carry `Field(description=...)` does not also get an `Attributes:` block — but fold anything the block says that the descriptions do not into the descriptions first. `tests/test_field_descriptions.py` enforces that, because the rule had already decayed once: 77 fields' two copies had drifted apart before anything scanned for them.

`AGENTS.md` (which `CLAUDE.md` links to) holds the rules the code at the edit site does not show. When a fact is written in the code, `AGENTS.md` carries the pointer to it rather than a second copy.

## Logging

Pick the level from how tolerable the failure is, not from how deep in the stack it happened. A routine, user-driven outcome such as a deleted message or a private remote post is not an error, and logging it as one makes real regressions unfindable.

| Level           | Use for                                                                                                                                                                                                                | Exception    |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `logfire.debug` | High-volume tracing that only helps while diagnosing one specific flow: per-attachment upload timings, per-tick internals, per-retry attempts. Never a failure on its own.                                             | optional     |
| `logfire.info`  | A normal lifecycle event, or a routine user-driven outcome that is not a defect: a message or thread the user deleted, a private or removed remote post, a cancelled task, a kill-switch or unconfigured-feature skip. | usually none |
| `logfire.warn`  | A degraded but handled outcome: the feature fell back, partially delivered, or dropped one item of many. A failed best-effort path belongs here.                                                                       | required     |
| `logfire.error` | An unexpected failure that broke a user-visible deliverable, left state inconsistent, or points at a defect someone must look at.                                                                                      | required     |

- Every log statement inside an `except` attaches the exception (`_exc_info=True`, or `_exc_info=exc` when the handler binds it), plus `error_type=type(exc).__name__` when the handler is broad. The one carve-out is a failure that is **expected rather than diagnosable**, where the exception's own type is the whole finding and the stack is identical on every occurrence: catch that type on its own, say what it means in the message, and attach nothing. A permission the bot never had and cannot earn is the example — `utils/message_cleanup.py`'s `Forbidden` branch, which is split from the `HTTPException` one precisely so a 5xx keeps its traceback. A broad handler never qualifies, because there the type is what you came to find out.
- Every log carries the structured fields that identify its subject (`message_id`, `url`, `scope`, `thread_id`, `filename`), so a recurrence is greppable without reading the traceback.
- A broad `except Exception` or `contextlib.suppress(Exception)` is allowed only as a deliberate best-effort boundary. When it is, a comment says why it stays broad, and the handler still logs. `services/memory/inflight.py::safe_db_write` is the reference shape.
- Silent swallowing is reserved for inert cleanup where a log would be pure noise, such as removing a reaction or deleting an already-deleted message.
- A coarse `except` spanning several distinct steps gets split so the message names the step that actually failed. Do not split when narrowing would let an exception escape into a listener or fire-and-forget task that cannot handle it; keep it broad and say so.
- `LOG_LEVEL` sets the console and log-file floor, defaulting to `debug` so `./data/logs` holds the full trace.
- Feature usage is not logged, it is recorded: one JSON line per slash invocation and per AI reply in `./data/usage/<YYYY-MM>.jsonl`, so an occasional stocktake can find the features nobody uses. Those records are outside `./data/logs` on purpose — that file is debug-level, hand-cleaned and gated on `LOG_LEVEL`, so a history kept inside it dies with it. They hold numeric ids plus the Discord username as a label, never message content or command arguments, and nothing prunes them. `USAGE_LOG_ENABLED=false` turns recording off.

## LLM And Media Paths

- Runtime LLM calls go through `AsyncOpenAI` clients and the OpenAI Responses API, apart from the direct-to-Google paths below.
- `OPENAI_BASE_URL` usually points at LiteLLM. Provider-specific behavior should be expressed through model names, `ModelSettings`, tools, or `extra_body`.
- Do not import a provider-native SDK such as `anthropic` into a request path. `google-genai` is allowed only on the direct-to-Google paths that [AGENTS.md](https://github.com/Mai0313/discordbot/blob/main/AGENTS.md#backend-and-invariants) lists. Development scripts may use any of them.
- Runtime model strings for `./src` live in `RuntimeModelCatalog` in `src/discordbot/typings/models.py`; update that catalog instead of hardcoding names at call sites.
- Preserve the reaction-based progress UX for AI replies. The bot should not send intermediate "thinking" messages there.
- A link expansion is the exception, and it takes video delivery's shape rather than a status message of its own: the cog replies with one subtext line as it starts and edits that same message into the finished card, so the card cannot drift away from the link while the post is being read. A failure deletes it and the reaction is the whole report. That placeholder is persisted, so a restart runs the interrupted expansion again instead of leaving a line that never resolves. It is also claimed before any reaction goes on, since reactions share a per-channel rate-limit bucket that a message send does not.
- An auto-expansion cog subclasses `ExpansionCog` (`src/discordbot/utils/expansion_cog.py`), which owns the listener, the reply slot, the restart sweep, the failure classification and one reaction vocabulary — ✅ delivered, ⏱️ refused or stalled and worth retrying, ⚠️ nothing showable in what the platform served, ❌ the bot broke. Which one a read failure earns comes off the exception's class (`src/discordbot/utils/link_errors.py`), so no cog decides it. A cog supplies its URL pattern, its constants, a `read` and a `build_delivery`, or, subclassing `ConversationExpansionCog` (a platform read in one blocking call), which writes both, its reader and the card's platform-specific hooks; only the card differs per platform. `tests/test_expansion_contract.py` holds every cog to that and fails until a new source is written into it.
- An expansion never posts a second message. Anything the card cannot carry is counted inside it — a follow-up reply lands wherever the channel has got to, which is exactly what the placeholder exists to prevent.
- Video delivery keeps progress text on the deferred original message, then edits that same message with the final file and source URL. A lone file too big to upload is posted as its hosted link alone, since a second URL stops Discord rendering the inline player.

## Long-Term Memory

The memory store's invariants live in the [Memory section of AGENTS.md](https://github.com/Mai0313/discordbot/blob/main/AGENTS.md#memory); read it before changing anything that reads or writes `data/memories`.

## Economy And Games

The ledger and casino invariants live in the [Economy](https://github.com/Mai0313/discordbot/blob/main/AGENTS.md#economy) and [Games](https://github.com/Mai0313/discordbot/blob/main/AGENTS.md#games) sections of AGENTS.md; read them before changing anything that moves 虛擬歡樂豆.

## Tests And Quality Gates

The pytest configuration lives in `pyproject.toml`.

```bash
uv run pytest
```

Coverage must stay at or above 80%. CI runs tests on Python 3.12 and 3.13 for pushes and pull requests targeting `main`, `master`, or `release/*`. What decides whether they run is the diff, never the branch name: the suite is skipped when every changed file is Markdown, and `src/discordbot/cogs/gen_reply/capabilities.md` and the three READMEs do not count as Markdown there because `tests/test_capabilities.py` reads them.

Async tests that record nested test-double calls in a list must compare a stable invariant such as a mapping, set, `Counter`, or sorted value. An exact sequence assertion needs an adjacent `# order-contract: <reason>` explaining the production guarantee. When completion order is the behavior under test, control it with an event or barrier instead of relying on scheduler timing.

The pre-commit gate is the canonical local quality check:

```bash
uvx pre-commit run -a
```

It runs Ruff formatting and linting, ty type checking, Markdown formatting, ShellCheck, codespell, gitleaks, uv lock checks, and standard file hygiene hooks.

## Documentation

Which document is for whom, and what must move together, lives in the [Documentation Split section of AGENTS.md](https://github.com/Mai0313/discordbot/blob/main/AGENTS.md#documentation-split). `CONTRIBUTING.md` itself is developer-facing and stays in English.

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

By contributing, you agree that your contribution is licensed under the [MIT License](https://github.com/Mai0313/discordbot/blob/main/LICENSE).
