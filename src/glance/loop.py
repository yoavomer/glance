"""The agent loop: orchestrator -> action -> environment -> perception -> text.

The screenshot goes from the environment straight into PerceptionAgent and is
never handed to the orchestrator. The only thing that flows back is an
Observation, which the orchestrator receives as rendered text.
"""

from dataclasses import dataclass

from glance.env import Environment, EnvironmentClosed
from glance.orchestrator import Orchestrator
from glance.perception import PerceptionAgent
from glance.trace import Trace
from glance.types import Observation

DEFAULT_CHECK = "What is on the screen now, and did anything change as a result of the last action?"
HISTORY_LINE_CHARS = 200


@dataclass
class RunResult:
    success: bool
    result: str
    steps: int
    trace_dir: str


def _fmt_action(tool: str, args: dict) -> str:
    shown = {k: v for k, v in args.items() if k != "check"}
    return f"{tool}(" + ", ".join(f"{k}={v!r}" for k, v in shown.items()) + ")"


def _clip(s: str) -> str:
    return s if len(s) <= HISTORY_LINE_CHARS else s[: HISTORY_LINE_CHARS - 1] + "…"


class AgentLoop:
    def __init__(self, env: Environment, orchestrator: Orchestrator | None = None,
                 perception: PerceptionAgent | None = None, max_steps: int = 20, verbose: bool = True):
        self.env = env
        self.orchestrator = orchestrator or Orchestrator()
        self.perception = perception or PerceptionAgent()
        self.max_steps = max_steps
        self.verbose = verbose

    def _observe(self, question: str, step: int, trace: Trace) -> Observation:
        meta = self.env.metadata()
        if meta.get("url", "about:blank") == "about:blank":
            # Nothing to look at; don't pay for a perception call.
            return Observation(summary="Empty browser tab; no page loaded yet.", elements=[], answer="", url="about:blank")
        png = self.env.screenshot()
        trace.screenshot(step, png)
        obs = self.perception.observe(png, question, self.env.width, self.env.height)
        obs.url, obs.title = meta.get("url", ""), meta.get("title", "")
        return obs

    def _act(self, tool: str, args: dict) -> None:
        env = self.env
        match tool:
            case "click": env.click(args["x"], args["y"])
            case "type": env.type(args["text"])
            case "key": env.key(args["name"])
            case "scroll": env.scroll(args["dx"], args["dy"])
            case "navigate": env.navigate(args["url"])
            case _: raise ValueError(f"unknown tool {tool!r}")

    def run(self, task: str) -> RunResult:
        trace = Trace()
        trace.write({"type": "start", "task": task, "orchestrator_model": self.orchestrator.model,
                     "perception_model": self.perception.model, "viewport": [self.env.width, self.env.height]})
        history: list[str] = []
        obs = self._observe(DEFAULT_CHECK, 0, trace)
        self._log(f"TASK: {task}\ntrace: {trace.dir}")
        try:
            for step in range(1, self.max_steps + 1):
                d = self.orchestrator.decide(task, history, obs, step, self.max_steps)
                action = _fmt_action(d.tool, d.args)
                self._log(f"\n[{step}] {d.thought}\n    -> {action}   "
                          f"(orchestrator context {d.usage['context_tokens']} tok)")

                record = {"type": "step", "step": step, "thought": d.thought, "tool": d.tool, "args": d.args,
                          "orchestrator_usage": d.usage}
                if d.tool == "done":
                    trace.write(record)
                    self._log(f"\nDONE success={d.args['success']}: {d.args['result']}")
                    return RunResult(d.args["success"], d.args["result"], step, str(trace.dir))

                error = None
                if d.tool == "look":
                    question = d.args["question"]
                else:
                    question = d.args.get("check") or DEFAULT_CHECK
                    try:
                        self._act(d.tool, d.args)
                    except Exception as e:  # bad coordinates, navigation errors, ...: report, don't crash
                        error = f"{type(e).__name__}: {e}"

                obs = self._observe(question, step, trace)
                if d.tool == "look":
                    outcome = f"A: {obs.answer}"
                elif error:
                    outcome = f"ERROR {error}"
                else:
                    outcome = obs.answer if d.args.get("check") else obs.summary
                history.append(_clip(f"{step}. {action} -> {outcome}"))

                record.update(error=error, question=question, observation=obs)
                trace.write(record)
                self._log(f"    eyes ({obs.usage.get('input_tokens', 0)} tok): {obs.answer or obs.summary}")

            self._log("\nStopped: max_steps reached")
            return RunResult(False, "max_steps reached", self.max_steps, str(trace.dir))
        except EnvironmentClosed as e:
            self._log(f"\nStopped: {e}")
            return RunResult(False, f"stopped: {e}", step, str(trace.dir))
        finally:
            trace.close()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
