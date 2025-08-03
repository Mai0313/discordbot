import aiohttp
import nextcord


async def send_to_webhook(url: str, content: str) -> None:
    async with aiohttp.ClientSession() as session:
        webhook = nextcord.Webhook.from_url(url, session=session)
        await webhook.send(content)
