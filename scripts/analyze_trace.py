"""Compare the orchestrator's real context size with a screenshot-in-context baseline.

"This design" is measured: the context_tokens the API reported for each
orchestrator call during the run.

"Baseline" is reconstructed exactly, not estimated: for each step we build
the transcript a conventional computer-use agent would carry at that point
(same task, same actions, and every screenshot fed back as a tool result)
and count it with the token-counting endpoint, which is free and runs no
model. Caveat: it replays THIS run's trajectory; a real baseline agent might
take a different path.

    python scripts/analyze_trace.py [runs/<dir>] [--svg docs/context.svg]
"""

import base64
import json
import sys
from pathlib import Path

import anthropic

from glance.config import settings
from glance.orchestrator import SYSTEM, TOOLS

# $/1M tokens (input side only; output tokens are similar for both designs).
OPUS_IN, OPUS_CACHE_READ, OPUS_CACHE_WRITE = 4.00, 0.20, 5.00
HAIKU_IN = 0.10


def load(run_dir: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in (run_dir / "trace.jsonl").open()]
    start = next(r for r in rows if r.get("type") == "start")
    return start, [r for r in rows if r.get("type") == "step"]


def image_block(path: Path) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.standard_b64encode(path.read_bytes()).decode()}}


def baseline_contexts(client: anthropic.Anthropic, run_dir: Path, task: str, steps: list[dict]) -> list[int]:
    """Context size a screenshot-feeding agent would have when making each decision."""
    messages: list[dict] = [{"role": "user", "content": task}]
    sizes = []
    for s in steps:
        sizes.append(client.messages.count_tokens(
            model=settings.orchestrator_model, system=SYSTEM, tools=TOOLS, messages=messages,
        ).input_tokens)
        # Then append what this step adds to a conventional transcript: the call and a screenshot result.
        tool_id = f"toolu_{s['step']:04d}"
        messages.append({"role": "assistant", "content": [
            {"type": "tool_use", "id": tool_id, "name": s["tool"], "input": {"thought": s["thought"], **s["args"]}}]})
        shot = run_dir / f"step{s['step']:02d}.png"
        result = [image_block(shot)] if shot.exists() else [{"type": "text", "text": "ok"}]
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": result}]})
    return sizes


def costs(steps: list[dict], baseline: list[int]) -> dict:
    ours = 0.0
    for s in steps:
        u = s["orchestrator_usage"]
        ours += (u["input_tokens"] * OPUS_IN + u["cache_read_input_tokens"] * OPUS_CACHE_READ
                 + u["cache_creation_input_tokens"] * OPUS_CACHE_WRITE) / 1e6
        ours += s.get("observation", {}).get("usage", {}).get("input_tokens", 0) * HAIKU_IN / 1e6
    no_cache = sum(baseline) * OPUS_IN / 1e6
    # Best case for the baseline: each step re-reads the previous context from cache, writes only the new part.
    cached, prev = 0.0, 0
    for c in baseline:
        cached += (prev * OPUS_CACHE_READ + (c - prev) * OPUS_CACHE_WRITE) / 1e6
        prev = c
    return {"ours": ours, "baseline_no_cache": no_cache, "baseline_ideal_cache": cached}


# --- chart -------------------------------------------------------------------

SVG_STYLE = """
  .surface { fill: #fcfcfb; }
  .grid { stroke: #e6e5e0; }
  .axis { stroke: #b8b7b0; }
  .t1 { fill: #0b0b0b; } .t2 { fill: #52514e; }
  .s1 { stroke: #2a78d6; } .f1 { fill: #2a78d6; }
  .s2 { stroke: #eb6834; } .f2 { fill: #eb6834; }
  .ring { stroke: #fcfcfb; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .grid { stroke: #2e2e2c; }
    .axis { stroke: #57564f; }
    .t1 { fill: #f0efec; } .t2 { fill: #c3c2b7; }
    .s1 { stroke: #3987e5; } .f1 { fill: #3987e5; }
    .s2 { stroke: #d95926; } .f2 { fill: #d95926; }
    .ring { stroke: #1a1a19; }
  }
  text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; }
"""


def nice_step(maxv: float) -> int:
    for step in (500, 1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000):
        if maxv / step <= 6:
            return step
    return 100000


def write_svg(path: Path, ours: list[int], baseline: list[int]) -> None:
    W, H = 720, 380
    L, R, T, B = 64, 150, 56, 48
    pw, ph = W - L - R, H - T - B
    n = len(ours)
    tick = nice_step(max(baseline + ours))
    ymax = tick * -(-max(baseline + ours) // tick)

    def x(i: int) -> float:
        return L + (pw * (i - 1) / (n - 1) if n > 1 else pw / 2)

    def y(v: float) -> float:
        return T + ph - ph * v / ymax

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           f'role="img" aria-label="Orchestrator context tokens per step: this design vs screenshot-in-context baseline">',
           f"<style>{SVG_STYLE}</style>",
           f'<rect class="surface" width="{W}" height="{H}" rx="8"/>',
           f'<text class="t1" x="{L}" y="26" font-size="15" font-weight="600">Context the decision model reads at each step</text>',
           f'<text class="t2" x="{L}" y="44" font-size="12">tokens per orchestrator call, same run, same screenshots</text>']
    for v in range(0, ymax + 1, tick):
        out.append(f'<line class="{"axis" if v == 0 else "grid"}" x1="{L}" x2="{L + pw}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke-width="1"/>')
        out.append(f'<text class="t2" x="{L - 8}" y="{y(v) + 4:.1f}" font-size="11" text-anchor="end">{v // 1000}k</text>')
    every = 1 if n <= 15 else 2
    for i in range(1, n + 1):
        if i == 1 or i == n or i % every == 0:
            out.append(f'<text class="t2" x="{x(i):.1f}" y="{T + ph + 18}" font-size="11" text-anchor="middle">{i}</text>')
    out.append(f'<text class="t2" x="{L + pw / 2}" y="{H - 10}" font-size="12" text-anchor="middle">step</text>')

    for series, cls, label in ((baseline, "2", "screenshots in context"), (ours, "1", "this design (text only)")):
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(series, start=1))
        out.append(f'<polyline class="s{cls}" points="{pts}" fill="none" stroke-width="2" stroke-linejoin="round"/>')
        for i, v in enumerate(series, start=1):
            out.append(f'<circle class="f{cls} ring" cx="{x(i):.1f}" cy="{y(v):.1f}" r="4" stroke-width="2"/>')
        # Direct label at the line end: colored mark carries identity, text stays in ink colors.
        ex, ey = x(n), y(series[-1])
        out.append(f'<rect class="f{cls}" x="{ex + 10}" y="{ey - 5}" width="10" height="10" rx="2"/>')
        out.append(f'<text class="t1" x="{ex + 26}" y="{ey - 2}" font-size="12" font-weight="600">{series[-1]:,}</text>')
        out.append(f'<text class="t2" x="{ex + 26}" y="{ey + 12}" font-size="11">{label}</text>')
    out.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    args = sys.argv[1:]
    svg = Path(args[args.index("--svg") + 1]) if "--svg" in args else None
    dirs = [a for a in args if not a.startswith("--") and a != str(svg)]
    run_dir = Path(dirs[0]) if dirs else sorted(p for p in Path("runs").iterdir() if (p / "trace.jsonl").exists())[-1]

    start, steps = load(run_dir)
    client = anthropic.Anthropic()
    baseline = baseline_contexts(client, run_dir, start["task"], steps)
    ours = [s["orchestrator_usage"]["context_tokens"] for s in steps]

    print(f"run: {run_dir}\ntask: {start['task']}\n")
    print("| step | action | this design: orchestrator context | baseline: screenshots in context | perception input (discarded) |")
    print("|---:|---|---:|---:|---:|")
    for s, o, b in zip(steps, ours, baseline):
        p = s.get("observation", {}).get("usage", {}).get("input_tokens", "")
        print(f"| {s['step']} | `{s['tool']}` | {o:,} | {b:,} | {p:,} |" if p != "" else
              f"| {s['step']} | `{s['tool']}` | {o:,} | {b:,} | – |")
    c = costs(steps, baseline)
    print(f"\nfinal step: {ours[-1]:,} vs {baseline[-1]:,} tokens ({baseline[-1] / ours[-1]:.1f}x)")
    print(f"growth per step (avg): {(ours[-1] - ours[0]) / max(len(ours) - 1, 1):,.0f} vs "
          f"{(baseline[-1] - baseline[0]) / max(len(baseline) - 1, 1):,.0f} tokens")
    print(f"input cost, this design (orchestrator + perception, actual): ${c['ours']:.4f}")
    print(f"input cost, baseline on {settings.orchestrator_model}, no caching:  ${c['baseline_no_cache']:.4f}")
    print(f"input cost, baseline on {settings.orchestrator_model}, ideal caching: ${c['baseline_ideal_cache']:.4f}")

    if svg:
        write_svg(svg, ours, baseline)
        print(f"\nchart: {svg}")


if __name__ == "__main__":
    main()
