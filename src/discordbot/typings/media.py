"""Media handed between the fetch, the upload and the render.

Each of these was a bare tuple, and each was unpacked far from where it was built. A pair whose
slots only a docstring distinguishes reads the same whichever way round it is, so swapping them
costs nothing at the call site and everything at the provider.
"""

from datetime import datetime

from pydantic import Field, BaseModel


class LoadedMedia(BaseModel):
    """Bytes ready to upload, with the MIME type that decides how they are sent."""

    data: bytes = Field(..., description="The media's bytes, already downscaled where that applies.")
    mime_type: str = Field(
        ...,
        description="The media's real MIME type, which the upload needs and the part does not carry.",
        examples=["image/jpeg", "video/mp4"],
    )


class UploadedFile(BaseModel):
    """A file the provider has accepted and will serve until it expires."""

    uri: str = Field(
        ..., description="The full provider uri a content part references the file by."
    )
    expires_at: datetime = Field(
        ...,
        description="When the provider drops the file, which is also how long a render may be reused.",
    )
