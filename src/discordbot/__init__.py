import logfire
from importlib.metadata import version

logfire.configure(send_to_logfire=False, scrubbing=False)
__version__ = version("discordbot")
