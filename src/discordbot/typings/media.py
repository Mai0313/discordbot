"""Media handed between the fetch, the upload and the render.

Each of these was a bare tuple, and each was unpacked far from where it was built. A pair whose
slots only a docstring distinguishes reads the same whichever way round it is, so swapping them
costs nothing at the call site and everything at the provider.
"""

from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_text_param import ResponseInputTextParam
from openai.types.responses.response_input_image_param import ResponseInputImageParam


class LoadedMedia(BaseModel):
    """Bytes ready to upload, with the MIME type that decides how they are sent."""

    data: bytes = Field(
        ..., description="The media's bytes, already downscaled where that applies."
    )
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


# What a rendered attachment can be. The Gemini answer model reads a Files-API handle (an
# `input_file` carrying a file uri); a model that cannot resolve that uri gets the content inlined
# per type instead: images as `input_image` base64, PDFs as `input_file` base64 file data, and
# text or code files as `input_text`.
type RenderedPart = ResponseInputTextParam | ResponseInputImageParam | ResponseInputFileParam


class RenderedAttachment(BaseModel):
    """One attachment as the answer model will see it, and how long that render may be reused."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    part: RenderedPart = Field(..., description="The content part spliced into the request.")
    expires_at: datetime = Field(
        ..., description="When the render stops being valid, which bounds the per-message cache."
    )


class PendingUploadRepoll(BaseModel):
    """What re-polling an in-flight upload concluded.

    Two fields rather than one optional, because "stop, there is nothing yet" and "carry on,
    there was never anything to adopt" are different answers that both carry no file.
    """

    handled: bool = Field(
        ..., description="True to stop here; False to fall through to a fresh upload."
    )
    uploaded: UploadedFile | None = Field(
        default=None, description="The adopted file, or None while it is still processing."
    )
