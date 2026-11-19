## Discord Cogs & Surface Rules

- **A Discord `400001 Access to file uploads has been limited` on a guild days old lifted by itself (#719)**: check the guild's age before touching code.
- **A feature that shares schema, concepts or many helpers with an existing cog extends that cog** (the loan commands live in `cogs/economy/cog.py`); only a low-coupling feature gets a new one. Do not add a status or query command just for symmetry; fold the information into an existing command's embed, as `/balance` does (a preference: `/credit status` and `/central_bank status` exist).
- **A shared slash-option vocabulary is a `Literal`, never a `StrEnum`** (`typings/video.py::VideoQuality` has why and the sites a new preset must reach). `SlashOption(default=...)` is typed `Any` upstream, so `ty` cannot check a default.
- **Aligned-width embeds use `utils.discord_embeds.embed_spacer_payload(..., target=)`**.
- **Content past embed limits is paginated, rendered to a PNG via `attachment://...`, or clipped with a visible notice (`utils/discord_embeds.py::clip_to_utf16_limit`); never truncate silently.**
- **Markdown headings render reliably only inside `embed.description`.**
- **Never hardcode a Discord attachment-size ceiling; read it through `utils/media_delivery.py::upload_limit_for(guild=...)`, also passing `interaction=` when the media goes out through one**.
- **Decide attach vs host vs drop only in `MediaDeliveryPlanner.plan`; keep the Discord send and the degradation policy at each site**. Pass `envelope_margin=MEDIA_ENVELOPE_MARGIN` for a combined attach or one riding with embeds, and keep every new site host-free under `MEDIA_HOSTING_ENABLED=false` (module docstring).
- **A Discord forward has empty `content` / `embeds` / `attachments`; its payload is in `message.snapshots`** (`gen_reply/input.py`).
- **`channel.history(before=..., oldest_first=True)` reverses each 100-message page on its own**, so past one page the order scrambles: fetch newest first and reverse the list yourself (`gen_reply/surface.py::TurnSurface.fetch_history`).
- **A slash command's first followup after a defer keeps the defer's flag**, so a public settlement after an ephemeral defer goes through `utils/interaction_responses.py::send_expiring_followup_after_private_defer`.

## Usage Records

- **Usage records (`utils/usage_log.py`, #429) cover slash invocations and AI replies only**: a research run gets no record of its own, since its launch is already a `reply.db` `research` row and the `/deep_research` or `QA` turn that started it is recorded as usual. `admin/usage_report.py` names only what was used, so a zero-use inventory comes from `gen_reply/capabilities.md`.
