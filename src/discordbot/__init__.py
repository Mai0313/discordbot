from importlib.metadata import version

import logfire

logfire.configure(send_to_logfire=False, scrubbing=False)
__version__ = version("discordbot")
