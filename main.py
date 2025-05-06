from src.bot import DiscordBot
from src.types.config import DiscordConfig

if __name__ == "__main__":
    discord_config = DiscordConfig()
    bot = DiscordBot()
    bot.run(token=discord_config.discord_bot_token)
