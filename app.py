"""Chainlit interface for the RAG Operator Guide."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Optional

import chainlit as cl

from rag_backend import answer_query


logger = logging.getLogger(__name__)


# Import directly from the backend
ANSWER_QUERY_ASYNC = cl.make_async(answer_query)



@cl.on_chat_start
async def on_chat_start() -> None:
    """Send a quick greeting when the session starts."""
    await cl.Message(
        content="RAG Operator Guide assistant ready. Ask a question about the station."
    ).send()


def _format_steps(steps: Iterable[object]) -> str:
    """Format steps as a numbered Markdown list."""
    lines = []
    for idx, step in enumerate(steps, start=1):
        if isinstance(step, dict):
            title = step.get("title")
            instruction = step.get("instruction")
            if title and instruction:
                text = f"**{title}** — {instruction}"
            else:
                text = instruction or title or json.dumps(step)
        else:
            text = str(step)

        lines.append(f"{idx}. {text}")

    return "\n".join(lines)


def _build_image_elements(images: Iterable[object]) -> list[cl.Image]:
    """Convert image references into Chainlit elements."""
    elements: list[cl.Image] = []

    for entry in images:
        path: Optional[str] = None
        name: Optional[str] = None

        if isinstance(entry, dict):
            path = entry.get("path") or entry.get("file") or entry.get("url")
            name = entry.get("name")
        elif entry:
            path = str(entry)

        if not path:
            continue

        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path.cwd() / resolved

        if not name:
            name = resolved.name or "step-image"

        if resolved.exists():
            elements.append(
                cl.Image(path=str(resolved), name=name, display="inline")
            )
        else:
            logger.warning("Image not found on disk: %s", resolved)

    return elements


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming user messages."""
    query = message.content.strip()
    if not query:
        await cl.Message(content="Please provide a question for the assistant.").send()
        return

    try:
        raw_response = await ANSWER_QUERY_ASYNC(query)
    except Exception as exc:  # noqa: BLE001
        logger.exception("answer_query raised an exception")
        await cl.Message(
            content=f"Sorry, something went wrong while processing your query:\n{exc}"
        ).send()
        return

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse JSON response: %s", exc)
        await cl.Message(
            content=(
                "I couldn't parse the assistant's structured response. "
                "Here is the raw output instead:\n\n"
                f"```\n{raw_response}\n```"
            )
        ).send()
        return

    steps = payload.get("steps") or []
    if not isinstance(steps, Iterable) or isinstance(steps, (str, bytes)):
        steps = [steps]
    else:
        steps = list(steps)

    content_lines = []
    query_title = payload.get("query") or query
    if query_title:
        content_lines.append(f"**Query:** {query_title}")

    if steps:
        content_lines.append(_format_steps(steps))
    else:
        content_lines.append("_No steps returned._")

    images_used = payload.get("images_used") or []
    if not isinstance(images_used, Iterable) or isinstance(images_used, (str, bytes)):
        images_used = [images_used]
    else:
        images_used = list(images_used)

    elements = _build_image_elements(images_used)

    await cl.Message(content="\n\n".join(content_lines), elements=elements).send()


if __name__ == "__main__":
    cl.run(port=8000, host="0.0.0.0")
