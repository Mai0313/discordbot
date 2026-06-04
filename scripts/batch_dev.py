"""Local prompt development helper for LiteLLM Responses API batches."""

import json
import time
from typing import TYPE_CHECKING, cast
from pathlib import Path
from collections.abc import Sequence

from openai import OpenAI
from rich.console import Console

from discordbot.typings.llm import LLMConfig
from discordbot.typings.models import ModelSettings
from discordbot.cogs._gen_reply.prompts import REPLY_PROMPT

if TYPE_CHECKING:
    from openai.types.batch import Batch

console = Console()
config = LLMConfig()

# Mirror the @property value in cogs/gen_reply.py. slow_model has a time-of-day
# dispatch in production (peak hours swap to gemini-flash-latest); for
# dev we pin to the off-peak default. Swap manually when testing peak behaviour.
SLOW_MODEL = ModelSettings(name="gemini-flash-latest", effort="low")

BATCH_ENDPOINT = "/v1/responses"
BATCH_COMPLETION_WINDOW = "24h"
BATCH_POLL_INTERVAL_SECONDS = 10
BATCH_TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}
DEV_END_USER_ID = "batch_dev"


def _response_input(user_prompt: str, file_id: str | None = None) -> list[dict[str, object]]:
    """Builds the Responses API input for one batch request."""
    content: list[dict[str, str]] = [{"type": "input_text", "text": user_prompt}]
    if file_id is not None:
        content.append({"type": "input_file", "file_id": file_id})
    return [{"role": "user", "content": content}]


def _response_body(user_prompt: str, file_id: str | None = None) -> dict[str, object]:
    """Builds the non-streaming Responses API body for one batch request."""
    return {
        "model": SLOW_MODEL.name,
        "instructions": REPLY_PROMPT,
        "input": _response_input(user_prompt=user_prompt, file_id=file_id),
        "reasoning": SLOW_MODEL.reasoning,
        "tools": SLOW_MODEL.tools,
        "service_tier": "auto",
        "mock_testing_fallbacks": False,
        "cache": {"no-cache": True},
    }


def _batch_request(
    custom_id: str, user_prompt: str, file_id: str | None = None
) -> dict[str, object]:
    """Builds one JSONL request line for the Batch API."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": _response_body(user_prompt=user_prompt, file_id=file_id),
    }


def _write_batch_requests(
    user_prompts: Sequence[str], request_path: Path, file_id: str | None = None
) -> None:
    """Writes the Batch API JSONL input file."""
    request_path.parent.mkdir(parents=True, exist_ok=True)
    with request_path.open("w", encoding="utf-8") as output_file:
        for index, user_prompt in enumerate(user_prompts, start=1):
            request = _batch_request(
                custom_id=f"prompt-{index}", user_prompt=user_prompt, file_id=file_id
            )
            output_file.write(json.dumps(request, ensure_ascii=False))
            output_file.write("\n")


def _upload_user_file(client: OpenAI, file_path: str | Path | None) -> str | None:
    """Uploads an optional prompt attachment and returns its file id."""
    if file_path is None:
        return None

    path = Path(file_path)
    with path.open("rb") as input_file:
        file = client.files.create(
            file=input_file,
            purpose="user_data",
            extra_headers={"x-litellm-end-user-id": DEV_END_USER_ID},
            extra_body={"target_model_names": SLOW_MODEL.name},
        )
    console.print(file.model_dump())
    return file.id


def _upload_batch_file(client: OpenAI, request_path: Path) -> str:
    """Uploads the JSONL batch request file and returns its file id."""
    with request_path.open("rb") as input_file:
        file = client.files.create(
            file=input_file,
            purpose="batch",
            extra_headers={"x-litellm-end-user-id": DEV_END_USER_ID},
        )
    console.print(file.model_dump())
    return file.id


def _request_counts(batch: "Batch") -> str:
    """Formats request counts for polling output."""
    if batch.request_counts is None:
        return ""
    return (
        f", total={batch.request_counts.total}"
        f", completed={batch.request_counts.completed}"
        f", failed={batch.request_counts.failed}"
    )


def _wait_for_batch(client: OpenAI, batch_id: str, poll_interval_seconds: int) -> "Batch":
    """Polls a batch until it reaches a terminal status."""
    batch = client.batches.retrieve(
        batch_id=batch_id, extra_headers={"x-litellm-end-user-id": DEV_END_USER_ID}
    )
    while batch.status not in BATCH_TERMINAL_STATUSES:
        console.print(f"Polling... Status: {batch.status}{_request_counts(batch)}")
        time.sleep(poll_interval_seconds)
        batch = client.batches.retrieve(
            batch_id=batch_id, extra_headers={"x-litellm-end-user-id": DEV_END_USER_ID}
        )
    return batch


def _download_batch_file(client: OpenAI, file_id: str, output_path: Path) -> None:
    """Downloads a completed batch output or error file."""
    response = client.files.content(
        file_id=file_id, extra_headers={"x-litellm-end-user-id": DEV_END_USER_ID}
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data=response.content)
    console.print(f"[green]Saved to {output_path}[/green]")


def _extract_output_text(response_body: object) -> str:
    """Extracts human-readable output text from a Responses API result body."""
    if not isinstance(response_body, dict):
        return ""

    output_text = response_body.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()

    chunks: list[str] = []
    output = response_body.get("output")
    if not isinstance(output, list):
        return ""

    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if part.get("type") == "output_text" and isinstance(text, str):
                chunks.append(text)
    return "".join(chunks).strip()


def print_batch_results(result_path: str | Path) -> None:
    """Prints completed Responses API batch results from a downloaded JSONL file."""
    path = Path(result_path)
    for line in path.read_text(encoding="utf-8").splitlines():
        result = cast("dict[str, object]", json.loads(line))
        custom_id = result.get("custom_id", "unknown")
        console.rule(str(custom_id))

        error = result.get("error")
        if error is not None:
            console.print(f"[red]{error}[/red]")
            continue

        response = result.get("response")
        if not isinstance(response, dict):
            console.print("[red]Missing response payload[/red]")
            continue

        status_code = response.get("status_code")
        body = response.get("body")
        if status_code != 200:
            console.print(f"[red]HTTP {status_code}[/red]")
            console.print(body)
            continue

        output_text = _extract_output_text(response_body=body)
        console.print(output_text or body)


def retrieve_batch(
    batch_id: str,
    *,
    result_path: str | Path = "batch_dev_results.jsonl",
    error_path: str | Path = "batch_dev_errors.jsonl",
    poll_interval_seconds: int = BATCH_POLL_INTERVAL_SECONDS,
) -> None:
    """Polls an existing batch and downloads its output or error files.

    Args:
        batch_id: Batch id returned by `submit_gen_reply_batch`.
        result_path: Path where the successful output JSONL should be written.
        error_path: Path where the error output JSONL should be written.
        poll_interval_seconds: Seconds to sleep between polling requests.

    Raises:
        RuntimeError: The batch reaches a terminal non-completed status.
    """
    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    batch = _wait_for_batch(
        client=client, batch_id=batch_id, poll_interval_seconds=poll_interval_seconds
    )
    console.print(batch.model_dump())

    if batch.error_file_id is not None:
        _download_batch_file(
            client=client, file_id=batch.error_file_id, output_path=Path(error_path)
        )

    if batch.status != "completed":
        raise RuntimeError(f"Batch job ended with status: {batch.status}")

    if batch.output_file_id is None:
        raise RuntimeError("Batch job completed without output_file_id")

    _download_batch_file(
        client=client, file_id=batch.output_file_id, output_path=Path(result_path)
    )
    print_batch_results(result_path=result_path)


def submit_gen_reply_batch(
    user_prompts: Sequence[str],
    file_path: str | Path | None = None,
    *,
    request_path: str | Path = "batch_dev_requests.jsonl",
) -> str:
    """Submits a Responses API batch using the dev reply prompt.

    Args:
        user_prompts: User prompts to send as separate batch requests.
        file_path: Optional file to include as an input_file part in every request.
        request_path: Path where the generated JSONL request file should be written.

    Returns:
        The created batch id.
    """
    if not user_prompts:
        raise ValueError("user_prompts must contain at least one prompt")

    client = OpenAI(base_url=config.base_url, api_key=config.api_key)
    start = time.time()

    uploaded_file_id = _upload_user_file(client=client, file_path=file_path)
    resolved_request_path = Path(request_path)
    _write_batch_requests(
        user_prompts=user_prompts, request_path=resolved_request_path, file_id=uploaded_file_id
    )
    console.print(f"[green]Saved to {resolved_request_path}[/green]")

    batch_file_id = _upload_batch_file(client=client, request_path=resolved_request_path)
    batch = client.batches.create(
        input_file_id=batch_file_id,
        endpoint=BATCH_ENDPOINT,
        completion_window=BATCH_COMPLETION_WINDOW,
        metadata={"source": DEV_END_USER_ID, "model": SLOW_MODEL.name},
        extra_headers={"x-litellm-end-user-id": DEV_END_USER_ID},
    )

    end = time.time()
    console.print(batch.model_dump())
    console.print(
        f"\nSubmitted {batch.id} to {SLOW_MODEL.name} on Litellm in {end - start:.2f} seconds"
    )
    return batch.id


def gen_reply_batch(  # noqa: PLR0913 -- dev helper keeps file paths explicit at call sites.
    user_prompts: Sequence[str],
    file_path: str | Path | None = None,
    *,
    request_path: str | Path = "batch_dev_requests.jsonl",
    result_path: str | Path = "batch_dev_results.jsonl",
    error_path: str | Path = "batch_dev_errors.jsonl",
    wait: bool = False,
) -> str:
    """Submits a dev reply batch and optionally waits for the result.

    Args:
        user_prompts: User prompts to send as separate batch requests.
        file_path: Optional file to include as an input_file part in every request.
        request_path: Path where the generated JSONL request file should be written.
        result_path: Path where the successful output JSONL should be written when waiting.
        error_path: Path where the error output JSONL should be written when waiting.
        wait: Whether to poll until the batch reaches a terminal state.

    Returns:
        The created batch id.
    """
    batch_id = submit_gen_reply_batch(
        user_prompts=user_prompts, file_path=file_path, request_path=request_path
    )
    if wait:
        retrieve_batch(batch_id=batch_id, result_path=result_path, error_path=error_path)
    return batch_id


if __name__ == "__main__":
    gen_reply_batch(
        user_prompts=["為何 37 是質數?", "用一句話說明 Batch API 適合什麼任務。"], wait=False
    )
    # retrieve_batch(batch_id="batch_...")
