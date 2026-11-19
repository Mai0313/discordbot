"""Media payloads passed between fetch, provider upload, and request rendering."""

from datetime import datetime

from pydantic import Field, BaseModel
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
        description=(
            "The media's real MIME type, which the upload needs and the part does not carry."
        ),
        examples=["image/jpeg", "video/mp4"],
    )


class UploadedFile(BaseModel):
    """A file the provider has accepted and will serve until it expires."""

    uri: str = Field(
        ...,
        description=(
            "The handle a content part references the file by: a full uri where the provider "
            "serves one, a bare file id elsewhere."
        ),
    )
    expires_at: datetime = Field(
        ...,
        description=(
            "When the provider drops the file, which is also how long a render may be reused."
        ),
    )


# Files-API handles use `input_file` with a file URI; inlined attachments use `input_image`
# (images), `input_file` (PDFs), or `input_text` (text/code).
type RenderedPart = ResponseInputTextParam | ResponseInputImageParam | ResponseInputFileParam


class RenderedAttachment(BaseModel):
    """One attachment as the answer model will see it, and how long that render may be reused."""

    part: RenderedPart = Field(..., description="The content part spliced into the request.")
    expires_at: datetime = Field(
        ..., description="When the render stops being valid, which bounds the per-message cache."
    )
