"""Prompts for the deep-research agent, kept apart from the cog that composes and dispatches them.

Two constants live here, and neither is inert text. `RESEARCH_SYSTEM_INSTRUCTION` is handed to the
managed agent as the native Interactions API `system_instruction`, and `THREAD_TITLE_PROMPT` is the
Responses API `instructions` of the short `fast_model` side call that names the thread. Both carry
system / developer authority over anything in the input, so editing a line here changes model
behavior directly and is never a refactor.

The agent prompt is the whole of what the agent is told: report shape, citation discipline, and the
say-so-plainly-instead-of-inventing rule for a figure it cannot source. There is exactly one tier
and one report, so nothing here varies per run — the only per-run addition is today's date, which
`cog.py::_system_instruction` appends rather than baking a stale date into this file. The title
prompt steers brevity in words instead of with a token cap, and `cog.py::_generate_thread_name`
still trims the answer to Discord's hard name limit and falls back to the brief's first line when
the call fails or times out.

Authored in English per project convention; the report language is steered at runtime by the
in-prompt "respond in the user's language" rule, not by writing the prompt in another language.
What teaches the QA answer model to ASK for research is `DEEP_RESEARCH_INSTRUCTION` in
`gen_reply/prompts.py`; this file only instructs a run that has already been launched.
"""

# `cog.py::_system_instruction` appends today's date to this before the agent sees it.
RESEARCH_SYSTEM_INSTRUCTION = """You are a thorough research analyst working inside a Discord bot.
You run long, autonomous, multi-source research and produce a well-structured, cited report.

CRITICAL — language: write the final report in the SAME language the user used in their request.
If the request is in Traditional Chinese, write everything in Traditional Chinese. Never switch
to English unless the user did.

Report quality:
- Use clear markdown headings, and comparison tables where they help.
- Ground every non-obvious claim in a source and keep the inline citations the tools provide.
- If a specific figure is unavailable, say so plainly instead of inventing one.
- Be comprehensive but readable: lead with the key findings.

The report is a clean analyst report, not casual chatter."""


# Instructions for the `fast_model` side call in `cog.py::_generate_thread_name`.
THREAD_TITLE_PROMPT = """Write a very short Discord thread title for the user's research request.
Output ONLY the title text: a handful of words (aim for well under ~10), in the SAME language as
the request, with no surrounding quotes, no trailing punctuation, and no labels or explanation."""
