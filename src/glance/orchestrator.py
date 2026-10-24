"""The orchestrator: decides the next action from text alone. Never sees an image.

Each step is a fresh single-turn request, rebuilt from scratch:
    task + one line per past step + the latest Observation in full.
It is NOT a growing chat transcript. A transcript would grow with every
tool_use/tool_result pair; this prompt grows by one short history line per
step and is otherwise constant in size.
"""

import time
from dataclasses import dataclass

import anthropic

from glance.config import settings
from glance.types import Observation

SYSTEM = """You control a web browser to complete a task for a user. You cannot see the \
screen. A vision assistant looks at each screenshot for you and reports what it shows: \
a summary, a direct answer to your question, and interactive elements with their \
click coordinates.

Every turn, call exactly one tool.

How to work:
- Click only coordinates that the vision assistant reported. Never invent coordinates.
- To enter text, click the input first, then type. Press Enter with the key tool to submit.
- Every action tool takes a `check` question. The vision assistant answers it from the \
screenshot taken right after your action, so ask what you need to know next \
(e.g. "Did the results load? What is the first result's title?"). Pass "" for a generic \
description.
- Use `look` when you need to know something about the current screen without acting.
- The vision assistant can be wrong. If an action seemed to have no effect, check \
before repeating it; don't repeat the same failing action more than twice.
- When the task is complete, call `done` with the result the user asked for. If it \
cannot be completed, call `done` with success=false and say why.
- Keep `thought` to one or two sentences."""


def _tool(name: str, description: str, props: dict, check: bool = True) -> dict:
    properties = {"thought": {"type": "string", "description": "Brief reasoning for this step."}}
    properties.update(props)
    if check:
        properties["check"] = {
            "type": "string",
            "description": "Question for the vision assistant about the screen after this action, or \"\".",
        }
    return {
        "name": name,
        "description": description,
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


INT = {"type": "integer"}
STR = {"type": "string"}

TOOLS = [
    _tool("look", "Ask the vision assistant a question about the current screen.", {"question": STR}, check=False),
    _tool("click", "Left-click at viewport coordinates.", {"x": INT, "y": INT}),
    _tool("type", "Type text into the focused element.", {"text": STR}),
    _tool("key", 'Press a key or chord, e.g. "Enter", "Tab", "Control+A".', {"name": STR}),
    _tool("scroll", "Scroll the page by pixels. Positive dy scrolls down.", {"dx": INT, "dy": INT}),
    _tool("navigate", "Go to a URL.", {"url": STR}),
    _tool(
        "done",
        "Finish the task.",
        {"success": {"type": "boolean"}, "result": {"type": "string", "description": "The answer or outcome for the user."}},
        check=False,
    ),
]


@dataclass
class Decision:
    tool: str
    args: dict  # tool input minus `thought`
    thought: str
    usage: dict


def render_prompt(task: str, history: list[str], obs: Observation, step: int, max_steps: int, note: str = "") -> str:
    parts = [f"TASK: {task}", f"STEP: {step} of {max_steps}"]
    parts.append("HISTORY:\n" + ("\n".join(history) if history else "(none yet)"))
    parts.append("CURRENT SCREEN:\n" + obs.render())
    if note:
        parts.append(note)
    return "\n\n".join(parts)


class Orchestrator:
    def __init__(self, model: str = settings.orchestrator_model, effort: str = "medium",
                 client: anthropic.Anthropic | None = None):
        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic()

    def decide(self, task: str, history: list[str], obs: Observation, step: int, max_steps: int) -> Decision:
        note = ""
        usage = {"input_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                 "output_tokens": 0, "calls": 0}
        start = time.monotonic()
        for _ in range(3):  # tool_choice "auto" doesn't guarantee a call; re-ask if none came back
            prompt = render_prompt(task, history, obs, step, max_steps, note)
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                # System prompt + tools are identical every step: cache them.
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS,
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": prompt}],
            )
            u = response.usage
            usage["input_tokens"] += u.input_tokens
            usage["cache_read_input_tokens"] += u.cache_read_input_tokens or 0
            usage["cache_creation_input_tokens"] += u.cache_creation_input_tokens or 0
            usage["output_tokens"] += u.output_tokens
            usage["calls"] += 1
            # Context size of the final successful call: what the model actually had to read.
            usage["context_tokens"] = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)

            if response.stop_reason == "refusal":
                raise RuntimeError(f"orchestrator refused: {response.stop_details}")
            call = next((b for b in response.content if b.type == "tool_use"), None)
            if call is not None:
                args = dict(call.input)
                thought = args.pop("thought", "")
                usage["latency_ms"] = int((time.monotonic() - start) * 1000)
                return Decision(tool=call.name, args=args, thought=thought, usage=usage)
            note = "Your previous reply did not call a tool. Call exactly one tool now."
        raise RuntimeError("orchestrator did not call a tool after 3 attempts")
