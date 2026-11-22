---
name: reply-quality-review
description: >-
  Diagnose Discord bot reply quality from data/database/reply.db feedback, data/database/messages.db conversation scenes, user/server memory files, and ./data/logs before proposing prompt, memory, or model changes. Use when reviewing thumbs-up/down feedback, investigating downvotes or verbosity complaints, evaluating blind-test cohorts, or deciding whether src/discordbot/cogs/gen_reply/prompts.py should change.
---

# Reply Quality Review

Produce an evidence-backed diagnosis of recent bot replies from live feedback and runtime artifacts, then discuss concrete Q&A examples with the user before editing `src/discordbot/cogs/gen_reply/prompts.py`, memory files, or model settings.

## Outcome

A completed review delivers:

- Overall and per-model approval rates (distinguishing all votes from asker-only votes), regeneration rates (`Regen`), pairwise head-to-head regeneration outcomes (`Pairwise`), and downvote reason breakdown for the inspected window.
- Average reply length by `(model, effort)` on `reply_turn` so verbosity patterns are measured rather than guessed.
- A classification of real defects into the four root-cause buckets below, with noisy votes separated out and explained.
- Whether an active slow-model blind test (`SLOW_MODEL_BLIND_TEST_ENABLED`) is running and would be confounded by a prompt edit.
- Real Q&A examples (with preceding channel context, voter identity, reason, and relevant memory/log findings) alongside concrete options to discuss before making any change.

## Gathering Evidence

Run commands from the repo root. Read SQLite files read-only so live WAL writers are never blocked.

1. **Summarize ratings and filter by cohort** with `uv run feedback-report`:

```bash
uv run feedback-report
uv run feedback-report --asker-only
uv run feedback-report --down-only
uv run feedback-report --inspect --down-only --context 5
```

Add `--since YYYY-MM-DD`, `--model <name>`, `--route QA`, or `--reason <tag>` (`off_topic`, `inaccurate`, `tone`, `verbose`, `other`) when narrowing to a deployment window, model, or complaint category. `--inspect` and `--show <reply_message_id>` automatically pull preceding messages from `data/database/messages.db` (degrading cleanly when absent or for `/ask` turns) and display `regen_of` / `regenerated_to` chain links when present.

Gateway and DM QA replies also carry a `🔄` regeneration button (`cogs/gen_reply/regen.py::ReplyRegenerator`): when the original asker (or bot owner) regenerates a reply and the replacement lands, the superseded turn receives an implicit downvote with `reason_tag='regen'` (preserving any earlier explicit `👎`) and the replacement records `regen_of_reply_message_id`. `uv run feedback-report` surfaces this through the overview `regens` row, the `Regen` rate column in `By Model`, and the `Pairwise` head-to-head table (`From -> To`, `Pairs`, `Explicit Win`, `Soft Accept`, `Both Failed`, `Accept Rate`). Note that `reason_tag='regen'` rows are excluded from `By Model` vote counts and approval rates (so regenerations are not double-counted beside `Regen`), yet they ARE included in `Recent Feedback`, `--down-only`, and `--inspect` outputs, and `--reason` accepts only the five explicit tags above so `regen` cannot be filtered via `--reason`.

2. **Measure verbosity by `(model, effort)`** directly on `data/database/reply.db` (read-only):

```bash
sqlite3 "file:data/database/reply.db?mode=ro" \
    "SELECT model, effort, COUNT(*) AS turns, ROUND(AVG(LENGTH(answer)), 1) AS avg_chars, MAX(LENGTH(answer)) AS max_chars FROM reply_turn WHERE route = 'QA' GROUP BY model, effort ORDER BY model, effort;"
```

Add `AND created_at >= 'YYYY-MM-DD'` when scoped to a window. Comparing `avg_chars` across models at the same `effort` separates a model's natural verbosity from a prompt-wide issue.

3. **Inspect injected memory and tone files** whenever a reply sounds oddly harsh, overly familiar, off-persona, or brings up stale/unasked facts:

- Per-user tone note: `data/memories/<author_id>/tone.md` (compare against `data/memories/<author_id>/detail.md` if the tone note looks suspiciously skewed).
- Per-user consolidated facts: `data/memories/<author_id>/global/*.md`, `data/memories/<author_id>/g/<guild_id>/*.md`, and `data/memories/<author_id>/dm/*.md`.
- Per-server community memory: `data/memories/bot_memories/<guild_id>/global/*.md`.

A reply that matches a user's `tone.md` or server banter norms is a memory/tone issue first, not a `prompts.py` bug.

4. **Correlate runtime execution in `./data/logs/`** using `source_message_id` from the scene card:

- Search `./data/logs/` for `message_id=<source_message_id>` (or the raw numeric id).
- Compare `gen_reply answer dispatch` (`model`, `effort`, `backend`, `history`, `reference`, `link_blocks`, `media_parts`, `user_memory`, `server_memory`, `tone`) against `gen_reply reply finalized` (`model`, `cost`, `web_search_queries`, `url_citations`, `reply_chars`, `attempts`). A different `model` on `finalized` than on `dispatch` indicates a LiteLLM proxy fallback.
- For attachment or link issues, check whether media parts or link blocks actually reached the answer turn before blaming the model.

## Filtering Noise Before Classifying

Do not treat every thumbs-down as a defect. Separate out:

- **Bystander / teasing clicks**: compare `--asker-only` against all votes; bystanders downvoting banter or another person's reply are weak signal unless the reason text names a real error.
- **Playful or meta downvotes**: read the preceding `messages.db` context and the voter's relationship to the conversation; users sometimes downvote a roast they invited or click buttons to test the UI.
- **Unreasoned one-offs**: a single thumbs-down with no reason tag on an otherwise accurate, well-scoped reply is low-confidence unless it fits a broader pattern.
- **Implicit regeneration downvotes (`reason_tag='regen'`)**: `--down-only` and `--inspect` include these rows; follow `regenerated_to` via `--show <child_id>` to compare the original and replacement answers and see whether the retry resolved the complaint (`Explicit Win` / `Soft Accept`) or whether both failed (`Both Failed`).
- **Transient upstream errors / retries**: check `attempts` and proxy fallbacks in `./data/logs/` before attributing an outage hiccup to the primary model or prompt.

## Root-Cause Classification

Assign each real finding to one primary bucket so the fix targets the right layer:

- **Model-specific**: concentrated in one model (or one `(model, effort)` cell) while other models on the same prompt behave well — e.g. one blind-test candidate has much higher `avg_chars` or hallucination rate. Candidate fixes: model selection, effort grading, or waiting for blind-test sample size rather than adding prompt rules that penalize every model.
- **Memory or tone**: driven by a stale, over-broad, or inverted entry in `data/memories/<author_id>/tone.md`, user compartment `*.md` facts, or `data/memories/bot_memories/<guild_id>/global/*.md`. Candidate fixes: memory/tone cleanup or memory-writer adjustment, not `prompts.py`.
- **Routing or tool**: wrong route (`QA` vs `IMAGE`/`VIDEO`), wrong `effort` tier, missed `watch_video` or `link_context_sources`, failed attachment/link ingest, missing web search, or proxy fallback. Candidate fixes: `route_prompt`, link/attachment handling, or tool gating.
- **Prompt-wide**: recurs across multiple models and users even with clean memory and intact context — e.g. formatting rules ignored, answer scope drifting to old history, or missing search discipline. Candidate fixes: targeted edits in `src/discordbot/cogs/gen_reply/prompts.py`.

## Safeguards Before Editing

- **Check for an active blind test (`SLOW_MODEL_BLIND_TEST_ENABLED`)**: inspect whether recent `reply_turn` rows distribute across the blind-test candidate models (or check the running deployment config without printing `.env` secrets). Editing `prompts.py` while a blind test is running confounds pre- and post-edit turns in the same comparison; warn the user explicitly and recommend either waiting for the blind test to finish or treating the edit date as a new `--since` cohort boundary.
- **Discuss real examples before editing `prompts.py`**: present the actual Q&A excerpts, context, and classification first, and get agreement on what should change. Every string in `prompts.py` is model input, and adding rules for single anecdotes degrades general replies.
- **Verify before deleting an existing prompt rule**: before proposing to remove a line from `prompts.py` as redundant, count whether the failure it guards against still appears in bot replies in `data/database/messages.db`.
