"""The deep-research cog.

`cog.py` owns the commands and the thread lifecycle; beside it sit the persistent session
store (`database.py`), the direct Gemini Interactions agent call layer (`agent.py`), the
escalation / plan-approval views (`views.py`), and the thread report delivery
(`delivery.py`).
"""
