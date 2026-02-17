FROM nikolaik/python-nodejs:python3.11-nodejs22 AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

LABEL maintainer="Wei Lee <mai@mai0313.com>" \
    org.label-schema.name="discordbot" \
    org.label-schema.vendor="Wei Lee" \
    org.label-schema.schema-version="1.0" \
    com.centurylinklabs.watchtower.stop-signal="SIGINT"

########################################################################################

FROM builder AS prod
WORKDIR /app
COPY . .
# System dependencies for media processing (yt-dlp merging requires ffmpeg)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN uv sync --no-dev && \
    uv cache clean
