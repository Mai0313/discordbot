# What I can do

## Talking to me

Mention me, reply to me, or send me a direct message. I hold a normal conversation, answer
questions, summarize what a channel has been talking about, read the images and files people
attach, and search the web whenever the answer could be out of date.

Somewhere I am not a member — a server I was never added to, a group DM, or your DMs with someone
else — none of that reaches me, because I never see those messages at all:

- `/ask` — say something to me there anyway, with a file attached if you want

It is the same conversation as anywhere else, and it remembers what we have already said to each
other in that channel, so you can pick up where you left off. What is missing there cannot be
added: I cannot read what anyone else typed, I keep no memory of that server's community, and my
usual reactions have nowhere to sit, so anything I had to drop is written under my reply instead.
A very long answer can also run out of room, and says so when it does.

I can also produce media as part of a reply when the moment calls for it: speak a line aloud,
draw one or more images, write and record a short song, or generate a short video from a
description or from attached images. Attach an image or a video (or reply to a message
carrying one) and I can edit that instead of making something new. All of this is asked for
in plain language; none of it has a command.

I slowly learn each person's preferences and how they like me to sound, and I may lean on
that in a reply. When I do, a 📖 note under my reply says whose memory I read. I decide what
is worth keeping while I answer, so you can also just tell me to remember something, or tell
me something I remember is wrong and should go; when I take a note, my reply says 正在整理記憶⋯
while I work on it, and that turns into a ✏️ line for what I took down or a 🩹 line for what I
dropped. Something I judged too personal to repeat in the channel is counted there rather than
quoted. I keep a separate memory of each server's community the same way, and what I remember
about you in one server does not follow you into another unless it is the kind of thing that is
safe anywhere.

### Links I open on my own

- **YouTube**: mention me with a YouTube link, or reply to a message carrying one, and I watch the video before answering.
- **Threads**: paste a Threads link on its own and I expand the post, its reply chain, the post it quotes, and its media into the channel. Mention me with the link instead and I read the post together with the comments underneath it, then answer about it.
- **Facebook**: paste a public Facebook post link on its own and I expand the post, its pictures and its counters into the channel; if the link carries a `comment_id`, I show that one comment under it as well. Mention me with the link instead and I read the post plus whichever comments the page loads up front, then answer about it. I can only read public posts, I cannot get at the video file on a video post, and I never see a whole comment section.
- **Instagram**: paste a public Instagram post link on its own and I expand the caption, its carousel pictures and its counters into the channel; a link that points at one comment shows that comment under it as well. Mention me with the link instead and I read the post together with its comments, then answer about it. I can only read public accounts, and I hand a Reel over as a link rather than watching it.
- **Twitter**: paste an x.com link on its own and I expand the post, its pictures, the post it replies to and the post it quotes into the channel. Mention me with the link instead and I read it and answer about it. Two things I never get: the replies underneath, where Twitter gives me a count and not one of them, so I cannot tell you what people said; and the whole of a long post, which reaches me cut off after its opening with the rest unavailable. I also hand a video over as a link rather than watching it.
- **Douyin**: paste a Douyin link on its own and I post the video, or the images for a photo post. Mention me with the link instead and I watch it and answer about it.
- **Bilibili**: mention me with a Bilibili video link and I watch it and answer about it.

A link that is only incidental to the question is left alone.

## Memory

- `/memory show` — what I remember about you
- `/memory regenerate` — rebuild my memory of you from scratch, in the background
- `/memory clear` — erase everything I remember about you
- `/memory server show` — what I remember about this server's community; only works inside a server I am a member of

## Research

- `/deep_research` — start a long, fully cited research report in its own thread; only works in an ordinary text channel of a server I am a member of, since it needs to open a thread, and only where that channel lets me post a message and open a thread from it; where it does not, I tell you I lack the permission. Asking me for deep research in conversation kicks off the same thing.

## Casino games

- `/games blackjack` — open a Blackjack table; a hand of five cards that has not busted wins outright
- `/games dragon_gate` — open a Dragon Gate table backed by a shared jackpot pool
- `/games blackjack_history` — recent Blackjack rounds, optionally for one member

## 虛擬歡樂豆 and the economy

Everyday:

- `/balance` — your cash, debts, net worth, and VIP status
- `/vip` — buy VIP, which boosts Blackjack payouts

Transfers and boards:

- `/give` — send 虛擬歡樂豆 to someone; 5% is burned as transfer tax
- `/leaderboard` — the wealth board
- `/loss_leaderboard` — today's biggest losers
- `/casino` — the casino system's own profit and loss
- `/pocat` — my wallet, since I sit at the tables as an ordinary player

Loans between members:

- `/credit borrow` — ask another member for a loan; the lender accepts or rejects it with a button, and it is rejected automatically after 180 seconds
- `/credit repay` — repay a lender
- `/credit call` — collect from someone who borrowed from you
- `/credit status` — every active personal contract you are on, as borrower or lender

Central bank:

- `/central_bank borrow` — request a loan from the central bank; a server administrator approves or rejects it with a button, and it is rejected automatically after 180 seconds
- `/central_bank repay` — repay your central-bank loan
- `/central_bank call` — server administrators only: forced collection, and only from someone who takes part in this server
- `/central_bank status` — how much the central bank can still lend in this server

The central bank has one lending budget shared by every server, first come first served. How
much of it a server can draw depends on how much the people who take part there hold — whoever
has been rewarded for talking there or has used a central bank command there — so a quiet server
can borrow less than a busy one, and once the budget is lent out nobody can borrow until some of
it is repaid. Balances are not per server either: one wallet follows you everywhere.

How much any one person may owe is capped separately, at twice what they own free and clear
minus what they already owe, counting every loan and not just central-bank ones. Borrowing
lowers that cap by the same amount, so repaying or earning more is what raises it again. A
request over the cap is refused before anyone is asked to approve it.

Central-bank loans are created out of nothing when approved and destroyed when repaid. The
interest on top is kept by the bank and lent out again, so the bank's budget grows as loans are
repaid with interest; `/central_bank status` shows it.

Balance maintenance:

- `/admin refund_tax` — economy admins only: add to someone's balance
- `/admin collect_tax` — economy admins only: take from someone's balance, never below zero

Economy admin is a flag on an account, set by whoever runs me: it is not a Discord role, being a
server admin does not grant it, and no command hands it out. Central bank approval is the other
way round — it is Discord's own administrator permission in the server the request was made in,
so it is never available in a DM.

## Telling the developer something

I have no command for this, and I cannot pass a message on myself. Whoever runs me keeps a contact
in my Discord profile description, so open my profile and read what it says there; that is where a
problem or a feature request goes. I do not know what that contact is — only that it is written
there — so read it rather than asking me for it.

## Tools

- `/clean_threads_url` — turn a Threads share link into the post's own URL, so passing it on no longer names whoever shared it; the answer is visible only to you and no media is fetched
- `/download_video` — download a video from a supported platform, a Douyin link included; a Douyin photo post comes back as images
- `/ping` — my response latency

## Adding me somewhere

Open my profile and there is an Add App button. It offers two choices: putting me in a server,
which needs someone who can manage that server, or adding me straight to your own account, which
nobody else has to approve. The same two choices sit behind
<https://discord.com/oauth2/authorize?client_id=1134904996178182225>.

On your own account I come with you: my slash commands work in every server you are in, in group
DMs, and in your DMs with other people, including servers I was never added to. That copy of me
is yours alone: nobody else in those places sees me there, or can use me through you.

Slash commands are the only part of me that travels that way. Mentioning me, replying to me, and
the links I expand on my own all need me to be a member of the server itself, so somewhere I have
not been added, `/ask` is how you talk to me instead. Two commands do not travel either:
`/deep_research` needs an ordinary text channel to open its thread in, and `/memory server show`
needs a server whose community memory I keep, so both refuse anywhere I am not a member.

`/ask` also runs on a clock. Discord closes a slash command fifteen minutes after you send it, so
if I am still making an image or a video when that runs out, I say so rather than going quiet.
Everything else I do finishes long inside it.

Only the person who added me can reach me in those places. In a group DM, someone who has not
added me cannot ask me anything through you.
