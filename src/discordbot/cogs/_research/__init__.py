"""Helper package for the deep-research cog (`cogs/research.py`).

Holds the persistent session store (`database.py`), the direct Gemini Interactions
agent call layer (`agent.py`), the escalation / plan-approval views (`views.py`),
and the thread report delivery (`delivery.py`). Kept under `_research/` so the
`cogs/*.py` loader glob does not pick it up as a cog.
"""
