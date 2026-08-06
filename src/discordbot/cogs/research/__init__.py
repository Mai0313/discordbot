"""The deep-research cog: a minutes-long cited report from a Gemini managed agent, in a thread.

Two entry points reach the same run. `/deep_research` is the explicit one; the implicit one is the
QA answer model emitting a `<deep-research>` brief, which `gen_reply` hands over by calling
`ResearchCogs.launch`. Either way the cog opens a thread off the asking message, paints the agent's
reasoning onto that thread's status message while it works, and posts the cited report there. That
one report is the whole feature: no tier to escalate into, no button under it, and one in-flight
research per owner. It only runs where a nested thread can exist, so a guild text channel and not a
DM, a thread, or a forum post.

`cog.py` owns the Discord surface and the thread lifecycle: the command, the `launch` hook, the
`on_ready` restart resume, and `is_research_thread`, which `gen_reply` reads so QA stays quiet
inside a thread this cog is still writing into. Beside it sit the persistent session store
(`database.py`, one row per thread in `reply.db`, serving as both the resume record and the
one-per-owner guard), the Interactions call layer (`agent.py`, holding the streaming run, the
re-attach on a dropped stream, and the terminal read that actually carries the report), the live
reasoning view (`streaming.py`), the report delivery into the thread (`delivery.py`), and the
agent's own prompts (`prompts.py`).

The agent work talks DIRECT to Google on `gemini_api_key` rather than through the LiteLLM proxy,
because a managed agent rides the native Interactions API; `store=True` is what lets a restart
re-attach to a run still going server-side. The small thread-title side call is the one thing here
that goes through the proxy, like any ordinary reply. Both entry points are gated on
`LLMConfig.deep_research_available`, so a deployment with the kill-switch off or no direct Gemini
key never opens a thread it cannot finish.

This file stays import-free on purpose. `_load_cogs_sync` only ever imports
`discordbot.cogs.research.cog`, and a re-exporting init would make
`from discordbot.cogs.research.database import ...` (which `tests/conftest.py` does) drag the whole
Discord surface in as a side effect of initialising the package.
"""
