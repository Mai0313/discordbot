from src.bot import DiscordBot
from src.types.config import Config

if __name__ == "__main__":
    config = Config()
    bot = DiscordBot()
    bot.run(token=config.discord_bot_token)
