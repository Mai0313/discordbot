"""Tests for economy command embeds."""

from datetime import UTC, datetime

from discordbot.typings.economy import LoanLenderType, LoanContractView, LoanContractStatus
from discordbot.cogs.economy.embeds import build_credit_status_embed

_EMBED_DESCRIPTION_LIMIT = 4096


def _contract(*, contract_id: int, lender_name: str = "lender") -> LoanContractView:
    """Builds one active personal contract owed by viewer 1 to `lender_name`."""
    opened_at = datetime(2026, 1, 1, tzinfo=UTC)
    return LoanContractView(
        contract_id=contract_id,
        lender_type=LoanLenderType.USER,
        lender_id=2,
        lender_name=lender_name,
        borrower_id=1,
        borrower_name="borrower",
        principal_remaining=10_000,
        interest_due=150,
        monthly_rate_bps=150,
        opened_at=opened_at,
        last_interest_accrued_at=opened_at,
        status=LoanContractStatus.ACTIVE,
    )


def test_credit_status_lists_every_contract_past_the_old_ten_cap() -> None:
    """An eleventh contract is on the embed, and nothing claims anything was held back."""
    contracts = [_contract(contract_id=index, lender_name=f"lender{index}") for index in range(11)]
    embed = build_credit_status_embed(contracts=contracts, viewer_id=1)
    description = embed.description or ""
    for contract in contracts:
        assert f"欠 {contract.lender_name} " in description
    assert "未顯示" not in description


def test_credit_status_reports_what_it_could_not_list() -> None:
    """A list past the description budget states its remainder instead of dropping it."""
    contracts = [
        _contract(contract_id=index, lender_name="長名字測試借款人帳號" * 3)
        for index in range(200)
    ]
    embed = build_credit_status_embed(contracts=contracts, viewer_id=1)
    description = embed.description or ""
    assert len(description) <= _EMBED_DESCRIPTION_LIMIT
    listed = sum(1 for line in description.split("\n") if line.startswith("欠 "))
    assert f"-# 還有 {len(contracts) - listed} 筆未顯示" in description


def test_credit_status_labels_the_side_the_viewer_is_on() -> None:
    """A contract the viewer lent on reads as owed to them, not by them."""
    embed = build_credit_status_embed(contracts=[_contract(contract_id=1)], viewer_id=2)
    assert "borrower 欠你 " in (embed.description or "")
