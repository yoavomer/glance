"""The perception sub-agent: the only component that ever sees a screenshot.

Each observe() call is a fresh, single-turn request: one image, one question,
no history. Whatever the model saw is gone once the call returns; only the
compact text Observation survives.
"""

import base64
import json
import time

import anthropic

from glance.config import settings
from glance.types import Element, Observation

MAX_ELEMENTS = 12

SYSTEM = f"""You are the eyes of a browser automation agent. The agent cannot see; \
it relies entirely on your description to decide where to click and what to type.

You get one screenshot and one question. Report:
- summary: 1-2 sentences on what the screen shows (what kind of page, its main state).
- answer: a direct answer to the question, based only on what is visible. If the \
answer is not visible, say so plainly; never guess.
- elements: up to {MAX_ELEMENTS} interactive elements (links, buttons, text inputs, \
checkboxes, tabs, menu items) most relevant to the question, most relevant first. \
Skip decorative or irrelevant ones.

For each element give x, y as the CENTER of the element in screenshot pixels, \
measured from the top-left corner. Clicking exactly there must hit the element, \
so be precise. Label is the element's visible text, or for an empty input its \
placeholder or purpose, under 60 characters."""

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "answer": {"type": "string"},
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "label": {"type": "string"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["role", "label", "x", "y"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "answer", "elements"],
    "additionalProperties": False,
}


class PerceptionAgent:
    def __init__(self, model: str = settings.perception_model, client: anthropic.Anthropic | None = None):
        self.model = model
        self.client = client or anthropic.Anthropic()

    def observe(self, screenshot: bytes, question: str, width: int, height: int) -> Observation:
        start = time.monotonic()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=SYSTEM,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.standard_b64encode(screenshot).decode(),
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Screenshot size: {width}x{height} pixels.\nQuestion: {question}",
                        },
                    ],
                }
            ],
        )
        if response.stop_reason != "end_turn":
            raise RuntimeError(f"perception call stopped with {response.stop_reason!r}")

        data = json.loads(next(b.text for b in response.content if b.type == "text"))
        elements = [
            Element(id=f"e{i}", role=e["role"], label=e["label"], x=e["x"], y=e["y"])
            for i, e in enumerate(data["elements"][:MAX_ELEMENTS], start=1)
            if 0 <= e["x"] < width and 0 <= e["y"] < height  # drop off-screen hallucinations
        ]
        return Observation(
            summary=data["summary"],
            elements=elements,
            answer=data["answer"],
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "latency_ms": int((time.monotonic() - start) * 1000),
            },
        )
