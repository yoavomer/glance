"""A longer, multi-part task used to measure context growth over more steps.

The README's context chart comes from this run (the demo task finishes in
~5 steps, too short to show a trend). Requires scrolling and two searches.

    python scripts/long_task.py [--headed]
"""

import sys

from glance.env import PlaywrightBrowser
from glance.loop import AgentLoop

TASK = (
    "Part 1: Go to en.wikipedia.org, search for \"Alan Turing\" using the search box, and open "
    "his article. Scroll down through the article (do not use the table of contents) until you "
    "reach the section about his early life and education, and find the name of the school he "
    "started attending at age 13. "
    "Part 2: Then use the search box again to open the article \"Ada Lovelace\" and find, in the "
    "infobox, the name of her father. "
    "Report both answers."
)

with PlaywrightBrowser(headless="--headed" not in sys.argv) as env:
    result = AgentLoop(env, max_steps=30).run(TASK)

print(f"\nsuccess={result.success} steps={result.steps}\nresult: {result.result}\ntrace: {result.trace_dir}")
