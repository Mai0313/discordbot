"""VIP command outcomes against the isolated ledger."""

from types import SimpleNamespace
import asyncio

import pytest

from discordbot.typings.economy import VIP_PURCHASE_COST, VipPurchaseResult
from discordbot.cogs.economy.cog import EconomyCogs
from discordbot.cogs.economy.embeds import (
    build_vip_already_embed,
    build_vip_success_embed,
    build_vip_insufficient_embed,
)
from discordbot.services.economy.database import buy_vip, get_vip, get_account, get_balance

from tests.helpers.casting import as_bot, as_interaction
from tests.helpers.economy import seed_balance
from tests.helpers.discord_mocks import FakeUser, FakeInteraction


@pytest.mark.parametrize("balance", [VIP_PURCHASE_COST, 120_000])
async def test_overlapping_vip_purchases_report_one_purchase(
    monkeypatch: pytest.MonkeyPatch, balance: int
) -> None:
    await seed_balance(user_id=1, name="alice", amount=balance)
    admitted = asyncio.Barrier(parties=2)
    purchase_lock = asyncio.Lock()

    async def overlapping_purchase(
        user_id: int, name: str, avatar_url: str
    ) -> VipPurchaseResult | None:
        # Both commands must reach the purchase before either can become VIP.
        await admitted.wait()
        async with purchase_lock:
            return await buy_vip(user_id=user_id, name=name, avatar_url=avatar_url)

    monkeypatch.setattr("discordbot.cogs.economy.cog.buy_vip", overlapping_purchase)
    cog = EconomyCogs(bot=as_bot(fake=SimpleNamespace()))
    interactions = [
        FakeInteraction(user=FakeUser(user_id=1, name="alice"), slash_command=True)
        for _ in range(2)
    ]
    async with asyncio.timeout(delay=5):
        await asyncio.gather(
            *(
                EconomyCogs.vip_command.callback(
                    self=cog, interaction=as_interaction(fake=interaction)
                )
                for interaction in interactions
            )
        )

    expected_titles = {
        build_vip_already_embed(actor_name="alice", avatar_url="").title,
        build_vip_success_embed(
            actor_name="alice",
            avatar_url="",
            result=VipPurchaseResult(
                new_balance=balance - VIP_PURCHASE_COST, cost=VIP_PURCHASE_COST
            ),
        ).title,
    }
    assert {
        interaction.followup.sent[0]["embed"].title for interaction in interactions
    } == expected_titles
    for interaction in interactions:
        assert interaction.response.deferred_ephemeral is True
        assert len(interaction.followup.sent) == 1
        assert interaction.followup.sent[0]["ephemeral"] is True
    assert await get_vip(user_id=1) is True
    account = await get_account(user_id=1)
    assert account is not None
    assert account.balance == balance - VIP_PURCHASE_COST
    assert account.total_spent == VIP_PURCHASE_COST
    assert account.total_earned - account.total_spent == account.balance


@pytest.mark.parametrize("balance", [0, VIP_PURCHASE_COST - 1])
async def test_vip_purchase_reports_an_insufficient_balance(balance: int) -> None:
    await seed_balance(user_id=1, name="alice", amount=balance)
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), slash_command=True)

    await EconomyCogs.vip_command.callback(
        self=EconomyCogs(bot=as_bot(fake=SimpleNamespace())),
        interaction=as_interaction(fake=interaction),
    )

    assert (
        interaction.followup.sent[0]["embed"].title
        == build_vip_insufficient_embed(
            actor_name="alice", avatar_url="", balance_now=balance
        ).title
    )
    assert await get_vip(user_id=1) is False
    assert await get_balance(user_id=1) == balance


async def test_existing_vip_is_not_charged_again() -> None:
    await seed_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    assert await buy_vip(user_id=1, name="alice") is not None
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), slash_command=True)

    await EconomyCogs.vip_command.callback(
        self=EconomyCogs(bot=as_bot(fake=SimpleNamespace())),
        interaction=as_interaction(fake=interaction),
    )

    assert (
        interaction.followup.sent[0]["embed"].title
        == build_vip_already_embed(actor_name="alice", avatar_url="").title
    )
    assert await get_balance(user_id=1) == 0


async def test_exhausted_vip_purchase_does_not_claim_insufficient_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await seed_balance(user_id=1, name="alice", amount=VIP_PURCHASE_COST)
    monkeypatch.setattr("discordbot.services.economy.database._CONDITIONAL_WRITE_MAX_RETRIES", 0)
    interaction = FakeInteraction(user=FakeUser(user_id=1, name="alice"), slash_command=True)

    await EconomyCogs.vip_command.callback(
        self=EconomyCogs(bot=as_bot(fake=SimpleNamespace())),
        interaction=as_interaction(fake=interaction),
    )

    assert interaction.followup.sent[0]["embed"].description == (
        "The purchase could not be completed. Please try again."
    )
    assert await get_vip(user_id=1) is False
    assert await get_balance(user_id=1) == VIP_PURCHASE_COST
