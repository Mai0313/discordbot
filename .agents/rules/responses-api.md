---
description: "Read before changing, planning or reviewing a model call or a measurement of one: cogs/gen_reply/, cogs/research/, cogs/memory/, services/memory/, the LLM and model typings, utils/llm*.py, scripts/*_dev.py, scripts/regen_memories.py, and the LLM utils' tests and input helper."
trigger: model_decision
paths:
  - "src/discordbot/cogs/gen_reply/**"
  - "src/discordbot/cogs/memory/**"
  - "src/discordbot/cogs/research/**"
  - "src/discordbot/services/memory/**"
  - "src/discordbot/typings/llm.py"
  - "src/discordbot/typings/models.py"
  - "src/discordbot/utils/llm*.py"
  - "scripts/*_dev.py"
  - "scripts/regen_memories.py"
  - "tests/utils/test_llm*.py"
  - "tests/helpers/llm_input.py"
---

## AI Pipeline

### Responses API Gotchas

- **Role and part type pair strictly**: `user` / `system` / `developer` take `input_*` parts, `assistant` only `output_text` or `refusal`.
- **Behavior rules go in `instructions` (or `COMMON_PROMPT`), never an input-side wrapper**: `instructions` has `developer` authority over all of `input` but does not persist across turns.
- **Separator messages use `role=system`, not `developer`**, for Gemini and Claude compatibility through LiteLLM.
- **Run a builtin-tools-only positive control before reading any zero as no grounding**: the Responses bridge reports it ONLY as `url_citation` annotations (`vertexaisearch.../grounding-api-redirect/` urls) and never as `groundingMetadata`, which chat completions and the native API do carry along with `webSearchQueries`.
- **A function tool can share a Gemini request with `googleSearch` / `urlContext`, but each surface needs something different** (measured live 2026-07: proxy on LiteLLM 1.93.0, native both direct and via the byte-transparent `/gemini` pass-through): the proxy needs `include_server_side_tool_invocations: true` flat in `extra_body`, without which `_resolve_search_tool_conflict` silently strips the search tools (Google itself 400s the unflagged mix), and with it grounding holds on Responses (streaming too) and chat completions, pro as on flash. `interactions.create` rejects the flag (`Unknown parameter 'tool_config'`) and needs none for `gemini-3.1-pro-preview`, `gemini-3.1-flash-lite`, `gemini-3.5-flash` and `gemini-3.6-flash`, while both `*-latest` aliases were refused, so suspect a bare alias wherever a capability gate reads the model string.
- **Re-measure the mix with a prompt nonce plus `{"cache": {"no-cache": true}}` in `extra_body`, never from LiteLLM source alone or from one sample**: the proxy's `cache: true` key ignores `extra_body`, so a flagged request replays an unflagged answer, and the drop warning is deduplicated, so it is no per-request signal.
- **A measurement through the proxy records `responses.model` on every sample and runs at concurrency 4 or less**: at 8, 16 of 24 `gemini-3.1-pro-preview` samples came back from the flash fallback with HTTP 200 (2026-09). Measure a route prompt or schema change by replaying logged turns (text from `messages.db` joined on `discord_message_id`), several runs per variant so run-to-run noise is the baseline, cache bypassed (#725).
- **A function call and a grounded answer never share a turn** (2026-07, N=20 per condition on `slow_model`, #409): the call ends the turn with empty text (145/145; whether a search ran on it is unmeasurable), and the next turn with the tool result grounds (19/20); streaming behaved the same, which #409 does not record.
- **Three measured traps before adding a function tool to a turn that searches**: no runtime call sends `extra_body`, so a tool without the flag silently ends grounding on every QA reply; merely offering it drew a call on pure-search questions 20/20, which restraint wording did not fix (10/14) and a sentence exempting live-external-fact questions did (0/14); and naming a tool in `instructions` without attaching it makes the model hallucinate the call (3/12), ending the turn.
- **Never put a mock-testing key in `extra_body`, whatever its value**: since LiteLLM 1.96 `proxy/route_llm_request.py::raise_if_mock_testing_params_disallowed` 400s on the name alone unless `general_settings.dangerously_allow_mock_testing_request_params` is true. The deployed proxy sets that opt-in (`~/repo/litellm/litellm.yaml`), so a mock key is not refused there: a falsy value does nothing and a truthy one injects a real synthetic failure into production traffic.
