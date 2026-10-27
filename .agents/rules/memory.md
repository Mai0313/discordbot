---
description: "Read before changing, planning or reviewing long-term memory: cogs/memory/, services/memory/, the memory typings and caps, gen_reply's cog, answer, context, markers and recall modules, utils/llm_transcript.py and utils/channel_visibility.py, scripts/regen_memories.py, their tests and memory helpers, or anything under data/memories/."
trigger: model_decision
paths:
  - "src/discordbot/cogs/memory/**"
  - "src/discordbot/services/memory/**"
  - "src/discordbot/typings/memory.py"
  - "src/discordbot/typings/context_budgets.py"
  - "src/discordbot/cogs/gen_reply/answer.py"
  - "src/discordbot/cogs/gen_reply/cog.py"
  - "src/discordbot/cogs/gen_reply/context.py"
  - "src/discordbot/cogs/gen_reply/markers.py"
  - "src/discordbot/cogs/gen_reply/recall.py"
  - "src/discordbot/utils/channel_visibility.py"
  - "src/discordbot/utils/llm_transcript.py"
  - "scripts/regen_memories.py"
  - "tests/cogs/memory/**"
  - "tests/services/memory/**"
  - "tests/cogs/gen_reply/test_answer.py"
  - "tests/cogs/gen_reply/test_cog.py"
  - "tests/cogs/gen_reply/test_context.py"
  - "tests/cogs/gen_reply/test_markers.py"
  - "tests/cogs/gen_reply/test_recall*.py"
  - "tests/cogs/gen_reply/test_streaming_memory_note.py"
  - "tests/scripts/test_regen_memories.py"
  - "tests/utils/test_llm_transcript.py"
  - "tests/helpers/memory.py"
  - "tests/helpers/server_memory.py"
  - "tests/helpers/user_memory.py"
  - "data/memories/**"
---

## Memory

- **The path is the privacy boundary** (#408; `store.py`'s docstring has the layout): do not move memory into SQLite.
- **The model never chooses a fact id or filename**: code mints it from compartment + summary (`facts.py::mint_fact_id`), which is what makes path traversal structurally impossible.
- **Keep the compartment boundary structural, not a prompt rule**: each consolidation call sees only the evidence routed to the compartment it writes (`consolidation.py::_consolidate_locked`). Information flows global -> guild, never guild -> global.
- **Pre-#464 prose `member_alias` rows are not backfilled**: `scripts/regen_memories.py` reaches a server scope only as a full evidence rebuild, and a row heals only when a consolidation delta rewrites it; a bare restatement just re-stamps it (`deltas.py::reconfirm_facts`).
- **Every write to the memory tree goes through `store.py`**: `read_memory_document`'s cache is exact only because each fact write through the store bumps the scope's generation (module docstring), so editing the tree out of process while the bot runs is unsupported.
- **The answer model decides what to remember, inline** (#596): the memory markers (see Inline markers in `reply-pipeline.md`) replaced the extraction pass, so do not reintroduce `MemoryWriterAI.extract` or the `PHASE1_PROMPT` / `SERVER_PHASE1_PROMPT` pair. The one exception is `/memory server catchup` (#1080): it reads a channel nobody addressed the bot in, so `propose_server_notes` stands in for the markers there, and its notes still take the same review (`services/memory/catchup.py` has why it bypasses the turn queue).
- **A forget is evidence, never a delete**: `<forget-memory>` writes a `### forget_request` into `raw.md` and consolidation emits the `delete` in a `deletes_only` call, so a forget can only remove a fact, no fact id reaches the answer prompt and the reply path grows no delete capability. `apply_deltas` drops anything but a `delete` in that call, so `PHASE2_PROMPT`'s FORGET REQUESTS block offers nothing else and a partly wrong fact is deleted whole (#869; `deltas.py::partition_forget_requests` has the routing). A delete it does apply also takes the deleted facts' evidence out of `raw.md` and `detail.md` by key (`deltas.py::drop_released_evidence`); never show a model forgotten evidence and rely on a prompt line not to use it (#730). A tone preference is never a fact, so `tone.py::forget_tone` answers the same forget in its own call whose output is line numbers only, which is what keeps it delete-only (#735).
- **A personal forget never reaches server memory, on purpose**: shared memory changing on one member's word is the problem, not a gap.
- **The 🩹 line under a reply is written at raw-append time and proves no forget**: the `Memory compartment consolidated` debug record's `deleted=` for that scope does (a tone forget never appears there). A hand edit under `data/memories` is folded into the bot's next commit for that scope (`git_history.py` adds the whole scope directory), so a deletion in a bot commit does not prove the bot made it.
- **A bad `tone.md` reads coherent on its own; compare it against `detail.md`**: `<existing_tone>` bullets carry no `evidence_kind`, so once a note has merged across batches only the never-invert rule protects a stated preference.
- **Inject the whole memory, never a shrunk summary** (#245): a reply reads memory once with no on-demand lookup, so nothing falls back on what the injection leaves out.
- **`inflight.py`'s `_pending_updates` slot takes no token guard**: a turn resumed at startup carries its old token (`gen_reply/cog.py::_resume_memory`) and can arrive behind a live turn's newer one, so keeping only the newer token drops what the resume rescues.
- **Do not add a Chinese pronoun/relation lexicon to the sharing gate** (`writer.py::_sanitize_observation`): #408 measured it re-locking 14.5% of all `global` observations.
- **Anti-injection is plumbing in `writer.py`, `facts.py` (the nickname table's render and parse) and `utils/llm_transcript.py::sanitize_identity`, not wording in `prompts.py`**; prompts treat conversation content as data, never instructions.
- **A pattern over memory text anchors on ASCII word boundaries (`re.ASCII` or an explicit `[A-Za-z0-9_]` lookaround), never Unicode `\b`/`\w`**: Python counts Chinese as a word character, so a name or token typed against it slipped the sharing gate (#894) and the secret scrub (#915) (`writer.py::_mentions_roster_name`).
- **Recall is deterministic first, the route's optional picks second** (`recall.py`'s docstring, `context.py::plan_recall`). Do not bring back a separate selector or a `get_user_memory` function tool: offered as a tool, the selector picked someone on 64% of turns that named nobody, while the same rules as a route field picked nobody on those turns (#725).
- **Neither note review sees stored memory.** The personal review gets a target-centered list (`writer.py::target_centered_memory_messages`) and the server review the whole `ReplyContext.message_list`; neither carries a memory block, so neither can re-ingest what is already stored.
- Every memory DB write but the clear's tombstone goes through best-effort `inflight.safe_db_write` (`services/memory/database.py` has the state machine).
- **`memory_job` gains no columns for the marker notes.** Nothing migrates this schema and `clear_job` is not best-effort, so a new column takes `/memory clear` down on a deployed bot; the notes ride inside `transcript` (`writer.py::render_turn_payload`).
- **Only `regeneration.py::regenerate_scope_memory`'s replace pass and a forget pass (`deletes_only`) get `allow_mass_delete=True`** (#970); the rebuild is shared by `/memory regenerate` and `scripts/regen_memories.py`, and do not describe the two as equivalent (`regeneration.py`'s docstring).
- **Every store write except the clear's serializes under `scope_lock(scope)`.** The clear skips the lock on purpose, so every file write must sit right after its `cleared_since` check with no `await` in between or an in-flight update resurrects what was cleared (`pipeline.py::clear_scope_memory`).
- **`scope` is the owner key and `compartment` the visibility axis; they are never the same word.**
- **Every clear goes through `pipeline.py::clear_scope_memory`**. A clear is whole only when every tier its docstring lists is neutralized, so an offline wipe with the bot stopped must also create a tombstone or remove that scope's row, and an online clear must use this protocol because staging writers can still exist.
- **A test reading a DEFERRED turn's `memory_job` row must first drain `inflight._db_tasks`** through `tests/helpers/memory.py::wait_for_persisted_writes`.
