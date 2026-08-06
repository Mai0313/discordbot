"""Shared test helpers: the surface ``tests/test_*.py`` calls into instead of re-deriving one.

Cog test modules each grew their own fake interaction / message families and their own
whole-string assertions over rendered copy. The fakes drifted apart, and the string checks failed
on wording and emoji refreshes that changed no behavior. What lives here:

- ``casting.py`` — the one ``cast`` per boundary between a plain test double and the real
  nextcord / google-genai type the production signature carries, and the inverse view that reads
  a typed Responses input back as plain dicts, plus the few builders whose real constructor wants
  more than a test has (``make_not_found``, ``make_stub_gemini_client``,
  ``make_media_hosting_config``).
- ``discord_mocks.py`` — the unified ``FakeUser`` / ``FakeResponse`` / ``FakeFollowup`` /
  ``FakeDiscordMessage`` / ``FakeInteraction`` doubles, recording what a cog sent rather than
  reaching Discord.
- ``economy_invariants.py`` — accounting-identity assertions read back through the real economy
  database helpers, returning the snapshot so a caller can layer its own exact check on top.
- ``embeds.py`` — structural embed assertions (a field by name, a title's category marker) that
  hand the field back, so a test pins the value encoding behavior and not the localized wording
  around it.
- ``llm_input.py`` — extractors that pull a named block back out of a recorded Responses
  ``input``, anchored on the production renderers themselves so a header rewording tracks instead
  of silently emptying an assertion.

Two constraints hold for every module in the package. The suite runs ``--doctest-modules`` over
``tests/``, so each one is imported at collection time and must not need runtime API credentials
(the Tests workflow provides none), and a ``>>>`` snippet in any docstring here would be collected
and executed as a real doctest, so none carries one. The other is the docstring convention: unlike
a ``tests/test_*.py`` file, whose parameters are pytest fixtures resolved by name, everything here
has real callers and documents its parameters in full. ``tests/test_docstrings.py`` enforces that
split, and this package is on the strict side of it.
"""
