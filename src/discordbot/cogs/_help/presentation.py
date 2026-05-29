"""Embed builders for the localized help view."""

from nextcord import Embed

from discordbot.cogs._help.content import HELP_COLOR, CATEGORY_ORDER, HelpGuide


def _apply_footer(embed: Embed, requester_name: str, requester_avatar_url: str) -> None:
    """Stamps the shared requester footer onto a help embed."""
    embed.set_footer(text=f"Requested by {requester_name}", icon_url=requester_avatar_url)


def build_overview_embed(
    guide: HelpGuide, requester_name: str, requester_avatar_url: str
) -> Embed:
    """Builds the landing overview embed with a one-line index per category."""
    lines = [guide.intro, ""]
    for key in CATEGORY_ORDER:
        section = guide.sections[key]
        lines.append(f"{section.emoji} **{section.label}** — {section.summary}")
    embed = Embed(title=guide.title, description="\n".join(lines), color=HELP_COLOR)
    _apply_footer(
        embed=embed, requester_name=requester_name, requester_avatar_url=requester_avatar_url
    )
    return embed


def build_section_embed(
    guide: HelpGuide, key: str, requester_name: str, requester_avatar_url: str
) -> Embed:
    """Builds the detail embed for one category."""
    section = guide.sections[key]
    embed = Embed(
        title=f"{section.emoji} {section.label}", description=section.detail, color=HELP_COLOR
    )
    _apply_footer(
        embed=embed, requester_name=requester_name, requester_avatar_url=requester_avatar_url
    )
    return embed
