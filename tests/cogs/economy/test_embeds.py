"""Tests for economy command embeds."""

from discordbot.cogs.economy.embeds import build_credit_status_embed
from discordbot.utils.discord_embeds import DISCORD_EMBED_DESCRIPTION_LIMIT

from tests.helpers.economy import personal_loan_contract


def test_credit_status_lists_an_eleventh_contract() -> None:
    """An eleventh contract is on the embed, and nothing claims anything was held back."""
    contracts = [
        personal_loan_contract(contract_id=index, lender_name=f"lender-{index:02d}")
        for index in range(11)
    ]
    embed = build_credit_status_embed(contracts=contracts, viewer_id=1)
    description = embed.description or ""
    for contract in contracts:
        assert contract.lender_name in description
    assert len(description.split("\n")) == len(contracts)


def test_credit_status_reports_what_it_could_not_list() -> None:
    """A list past the description budget states its remainder instead of dropping it."""
    lender_name = "長名字測試借款人帳號" * 3
    contracts = [
        personal_loan_contract(contract_id=index, lender_name=lender_name) for index in range(200)
    ]
    embed = build_credit_status_embed(contracts=contracts, viewer_id=1)
    description = embed.description or ""
    assert len(description) <= DISCORD_EMBED_DESCRIPTION_LIMIT
    lines = description.split("\n")
    listed = len(lines) - 1
    # The last line is the notice, not one more contract, and it counts what was held back.
    assert lender_name not in lines[-1]
    assert str(len(contracts) - listed) in lines[-1]


def test_credit_status_labels_the_side_the_viewer_is_on() -> None:
    """A contract the viewer lent on names who owes them; one they borrowed on, whom they owe."""
    contract = personal_loan_contract()
    as_lender = build_credit_status_embed(contracts=[contract], viewer_id=2).description or ""
    as_borrower = build_credit_status_embed(contracts=[contract], viewer_id=1).description or ""
    assert "alice" in as_lender
    assert "bob" not in as_lender
    assert "bob" in as_borrower
    assert "alice" not in as_borrower
