"""Tests for economy command embeds."""

from discordbot.cogs.economy.embeds import build_credit_status_embed
from discordbot.utils.discord_embeds import DISCORD_EMBED_DESCRIPTION_LIMIT

from tests.helpers.economy import personal_loan_contract


def test_credit_status_lists_an_eleventh_contract() -> None:
    """An eleventh contract is on the embed, and nothing claims anything was held back."""
    contracts = [
        personal_loan_contract(contract_id=index, lender_name=f"lender{index}")
        for index in range(11)
    ]
    embed = build_credit_status_embed(contracts=contracts, viewer_id=1)
    description = embed.description or ""
    for contract in contracts:
        assert f"欠 {contract.lender_name} " in description
    assert "未顯示" not in description


def test_credit_status_reports_what_it_could_not_list() -> None:
    """A list past the description budget states its remainder instead of dropping it."""
    contracts = [
        personal_loan_contract(contract_id=index, lender_name="長名字測試借款人帳號" * 3)
        for index in range(200)
    ]
    embed = build_credit_status_embed(contracts=contracts, viewer_id=1)
    description = embed.description or ""
    assert len(description) <= DISCORD_EMBED_DESCRIPTION_LIMIT
    listed = sum(1 for line in description.split("\n") if line.startswith("欠 "))
    assert f"-# 還有 {len(contracts) - listed} 筆未顯示" in description


def test_credit_status_labels_the_side_the_viewer_is_on() -> None:
    """A contract the viewer lent on reads as owed to them, not by them."""
    embed = build_credit_status_embed(contracts=[personal_loan_contract()], viewer_id=2)
    assert "alice 欠你 " in (embed.description or "")
