"""Shared bounds on how hard the bot may hit Douyin, plus the user-facing failure mapping.

Douyin's WAF bans a share path for tens of minutes once it is hit hard, and auto-expansion
turns every pasted link into a request rather than only the ones somebody ran a command for.
So request volume, not correctness, is the binding constraint here.

Three things hold it down, all cheap: the module-level payload and short-link caches in
`utils/douyin.py` (one fetch per post per window, none at all for a re-pasted link), the
per-URL lock below (simultaneous pastes of one link collapse into a single fetch instead of
racing past the cache), and the global semaphore (a burst of distinct links queues rather
than arriving at Douyin all at once).

Lives in the cog's helper package so both callers can share it: the expansion cog and
`gen_reply`'s context builder. `gen_reply` imports from here, never from the cog module.
"""

from discordbot.utils.douyin import DouyinBlockedError, DouyinTooLargeError, DouyinUnavailableError
from discordbot.utils.asyncio_locks import KeyedLockManager, LoopLocalSemaphore

# Concurrent Douyin fetches across every caller. Deliberately small: the cost of queueing a
# second link for a few seconds is nothing next to a WAF ban that outlasts it by minutes.
DOUYIN_FETCH_CONCURRENCY = 2

douyin_fetch_semaphore = LoopLocalSemaphore(capacity_provider=lambda: DOUYIN_FETCH_CONCURRENCY)

# Serializes work per pasted URL. The payload cache alone is not enough: two expansions of the
# same link that start together both miss it and both fetch.
douyin_url_locks: KeyedLockManager[str] = KeyedLockManager()


def douyin_failure_message(error: Exception) -> str:
    """Maps a Douyin failure to the message a user should see.

    A bot wall, a missing post, an oversize file and a stall are kept apart on purpose.
    Reporting any of them as a deleted post is the single worst outcome this feature can
    produce: it sends someone off to re-check a link that is perfectly fine. Only
    `DouyinUnavailableError` — Douyin explicitly filtering the post out — earns that wording.
    """
    if isinstance(error, DouyinUnavailableError):
        return "-# 這則貼文已被刪除或設為私人"
    if isinstance(error, DouyinBlockedError):
        return "-# 抖音暫時擋住了請求，請稍後再試"
    if isinstance(error, DouyinTooLargeError):
        return "-# 這支影片太大,沒有自動下載;需要的話可以用 `/download_video`"
    if isinstance(error, TimeoutError):
        return "-# 抖音回應太慢,這次沒有抓到;稍後再試一次"
    return "-# 檔案無法下載"
