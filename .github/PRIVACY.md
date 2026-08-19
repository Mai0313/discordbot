# Privacy Policy

**Last updated: 2026-08-05**

This policy describes how the Discord app **破貓** (application ID `1134904996178182225`), operated by Mai0313, handles data. The bot is self-hosted on hardware controlled by the operator; it is not a hosted service and has no company behind it.

This repository is also published as open source. If you run your own copy, you are the operator of that deployment and this policy does not cover it.

## What the bot collects

The bot only receives what Discord delivers for the servers it has been invited to and for direct messages sent to it.

| Data                                                                                              | Where it comes from                                                                                                                                 | Why                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Message content, author name and ID, channel name and ID, timestamps, attachment and sticker URLs | Messages from humans in channels the bot can see, direct messages to the bot, and the bot's own replies. Messages from other bots are not recorded. | Producing AI replies that follow the conversation, summarising recent chat, expanding links, and the message-triggered features.                                                        |
| A per-user long-term memory written in plain text                                                 | Derived by the AI from your own conversations with the bot                                                                                          | Letting the bot remember your preferences across conversations. It is scoped by where it was learned: something told in one server is not surfaced in another.                          |
| A per-server memory                                                                               | Derived from public channels only                                                                                                                   | Community context for replies in that server. Content that a server's `@everyone` cannot read never feeds it.                                                                           |
| Discord user ID, display name, avatar URL, and game state (balances, transactions, match history) | Your use of the economy and casino commands                                                                                                         | Running those features. The currency is virtual, has no monetary value, and no payment information is ever collected.                                                                   |
| Feature usage records containing numeric IDs only                                                 | Each slash command invocation and each AI reply                                                                                                     | Knowing which features are used. No names, no message content and no command arguments are stored in these records.                                                                     |
| Runtime diagnostic logs                                                                           | The bot process                                                                                                                                     | Debugging. These may contain excerpts of message content and stay on the operator's machine; telemetry is explicitly disabled and nothing is sent to an external observability service. |

Everything above is stored in local SQLite databases and local files on the operator's machine.

## Who the data is shared with

The bot is not a data broker. Nothing is sold, rented, or handed to advertisers. Data leaves the operator's machine in exactly three situations, all of them required to produce what you asked for:

- **AI providers.** To generate a reply, the message you sent, the recent conversation of that channel, and any attachment relevant to the request are sent to an OpenAI-compatible API endpoint (a LiteLLM proxy run by the operator, which routes to model providers such as Google). Attachments and linked media are uploaded to the Google Gemini Files API so the model can read them. Those providers process the request under their own terms.
- **Oversize file hosting.** A generated or downloaded file too large for Discord's upload limit is written to the operator's own static host under an unguessable, content-derived filename, and the link is posted in the channel. Anyone holding the link can open the file; it is not listed or indexed. These files are deleted automatically after 168 hours.
- **Discord.** Everything the bot posts goes back to Discord and is subject to Discord's own privacy policy.

## How long it is kept

- Hosted oversize files: deleted automatically after 168 hours.
- Your long-term memory: kept until you erase it with `/memory clear`, or until you ask the operator to remove it.
- Message records, game state and usage records: kept for as long as the bot runs, or until you ask for removal.
- Diagnostic logs: cleared periodically by the operator.

## Your choices

- `/memory show` shows everything the bot currently remembers about you, including where each fact can be used.
- `/memory clear` erases your stored memory. It asks for confirmation first.
- `/memory regenerate` rebuilds it from the underlying evidence.
- Removing the bot from a server stops all collection for that server.
- To request deletion of everything associated with your Discord user ID, or to ask what is held, email **mai@mai0313.com** with your Discord user ID. Requests are handled manually, normally within a few days.

## Children

The bot is not directed at children. Discord requires all users to meet its own minimum age requirement, and the bot is intended for users who meet it.

## Security

Data lives on hardware controlled by the operator, is not exposed publicly except for the oversize-file links described above, and is not replicated to any third-party storage service.

## Changes

This policy may change as the bot changes. The current version is always the one in this repository, and its history is public in the repository's commit log.

## Contact

Mai0313, **mai@mai0313.com**. Issues may also be opened at <https://github.com/Mai0313/discordbot/issues>.
