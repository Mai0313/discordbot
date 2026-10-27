"""Where Discord's markdown differs from standard markdown, as prompt text sent verbatim to a model.

Shared because the reply pipeline and deep research both post model-written markdown as Discord
messages and cannot import each other. A wording change here is a behavior change.
"""

# Nested under each prompt line that lets the model write markdown into a Discord message.
DISCORD_MARKDOWN_RULES = """
    * Discord renders only part of markdown: headings stop at `###` (`####` and deeper print as literal text); tables, horizontal rules (`---`), images, HTML, footnotes, task lists and LaTeX are not rendered; `__text__` is underline, not bold.
"""
