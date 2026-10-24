"""Milestone 2: is the perception sub-agent's output good enough to act on?

For each test page we take a screenshot, ask PerceptionAgent a question, then
grade its coordinates against the live DOM:
  - target: did it locate the one element the question is about, and does
    (x, y) fall inside that element's bounding box?
  - click accuracy: for EVERY element it reported, what is actually under
    (x, y)? A hit means the interactive element there has matching text.
The DOM is used only for grading; the agent never gets it.

    python scripts/smoke_perception.py [--runs N]
"""

import re
import sys
from pathlib import Path

from glance.env import PlaywrightBrowser
from glance.perception import PerceptionAgent

OUT = Path("runs/smoke_perception")

CASES = [
    {
        "name": "home",
        "url": "https://en.wikipedia.org/wiki/Main_Page",
        "question": "Where is the search box I should click to search Wikipedia?",
        "target": "input[name=search]",
    },
    {
        "name": "results",
        # fulltext=1 forces the results page instead of jumping to a matching article
        "url": "https://en.wikipedia.org/w/index.php?search=perception+context+agent&fulltext=1",
        "question": "What is the title of the first search result, and where is its link?",
        "target": ".mw-search-result-heading a",
    },
]

# Returns the interactive element under (x, y) and every name a human might call it.
UNDER_POINT_JS = """([x, y]) => {
  const sel = 'a, button, input, select, textarea, summary, label, [role=button], [role=link], [role=tab], [role=checkbox], [role=menuitem], [role=combobox], [role=searchbox]';
  const hit = document.elementFromPoint(x, y);
  const el = hit && hit.closest(sel);
  if (!el) return null;
  const names = [el.innerText, el.value, el.placeholder, el.getAttribute('aria-label'), el.title]
    .filter(Boolean).map(s => s.trim()).filter(Boolean);
  return {tag: el.tagName.toLowerCase(), names};
}"""


def words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def label_matches(label: str, names: list[str]) -> bool:
    lw = words(label)
    for name in names:
        nw = words(name)
        if not lw or not nw:
            continue
        if lw <= nw or nw <= lw or len(lw & nw) / len(lw) >= 0.5:
            return True
    return False


def grade(env: PlaywrightBrowser, case: dict, obs) -> dict:
    target = env.page.locator(case["target"]).first
    box = target.bounding_box()
    target_names = env.page.evaluate(
        "el => [el.innerText, el.placeholder, el.getAttribute('aria-label')].filter(Boolean)",
        target.element_handle(),
    )

    hits, details = 0, []
    for e in obs.elements:
        under = env.page.evaluate(UNDER_POINT_JS, [e.x, e.y])
        ok = bool(under) and label_matches(e.label, under["names"])
        hits += ok
        found = f"{under['tag']} {under['names'][:1]}" if under else "nothing interactive"
        details.append(f"    {'HIT ' if ok else 'MISS'} {e.render()}  -> {found}")

    # Target: any reported element whose point lies inside the target's box.
    in_box = [
        e for e in obs.elements
        if box["x"] <= e.x <= box["x"] + box["width"] and box["y"] <= e.y <= box["y"] + box["height"]
    ]
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    return {
        "hits": hits,
        "total": len(obs.elements),
        "target_hit": bool(in_box),
        "target_names": target_names,
        "target_center": (int(cx), int(cy)),
        "target_box": {k: int(v) for k, v in box.items()},
        "details": details,
    }


def main(runs: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    agent = PerceptionAgent()
    totals = {"hits": 0, "total": 0, "target_hits": 0, "cases": 0, "in": 0, "out": 0, "ms": 0}

    with PlaywrightBrowser() as env:
        for case in CASES:
            env.navigate(case["url"])
            shot = env.screenshot()
            (OUT / f"{case['name']}.png").write_bytes(shot)
            for run in range(1, runs + 1):
                obs = agent.observe(shot, case["question"], env.width, env.height)
                g = grade(env, case, obs)
                u = obs.usage
                print(f"\n=== {case['name']} (run {run}) — {u['input_tokens']} in / {u['output_tokens']} out tokens, {u['latency_ms']} ms")
                print(f"Q: {case['question']}")
                print(obs.render())
                print(f"  Grading against DOM:")
                print("\n".join(g["details"]))
                print(f"  click accuracy {g['hits']}/{g['total']}; "
                      f"target {g['target_names'][:1]} center {g['target_center']} box {g['target_box']}: "
                      f"{'HIT' if g['target_hit'] else 'MISS'}")
                totals["hits"] += g["hits"]
                totals["total"] += g["total"]
                totals["target_hits"] += g["target_hit"]
                totals["cases"] += 1
                totals["in"] += u["input_tokens"]
                totals["out"] += u["output_tokens"]
                totals["ms"] += u["latency_ms"]

    n = totals["cases"]
    print(f"\nSUMMARY over {n} calls: targets {totals['target_hits']}/{n}, "
          f"click accuracy {totals['hits']}/{totals['total']}, "
          f"avg {totals['in'] // n} in / {totals['out'] // n} out tokens, avg {totals['ms'] // n} ms")


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 1)
