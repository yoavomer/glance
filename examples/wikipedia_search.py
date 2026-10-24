"""Demo: search Wikipedia and extract the first result.

The orchestrator starts on an empty tab with only the task text. It has to
find the search box, search, and read the result back, seeing the page only
through the perception sub-agent's text.

    python examples/wikipedia_search.py [--headed] ["your query"]
"""

import sys
import time

from glance.env import PlaywrightBrowser
from glance.loop import AgentLoop

args = [a for a in sys.argv[1:] if a != "--headed"]
query = args[0] if args else "perception context agent"

TASK = (
    f"Go to wikipedia.org and search for \"{query}\" using the site's search box. "
    "On the search results page, report the title of the first result and the first "
    "sentence of its snippet."
)

with PlaywrightBrowser(headless="--headed" not in sys.argv) as env:
    result = AgentLoop(env, max_steps=15).run(TASK)
    if "--headed" in sys.argv:
        print("\n(closing the browser in 5 seconds)")
        time.sleep(5)

print(f"\nsuccess={result.success} steps={result.steps}\nresult: {result.result}\ntrace: {result.trace_dir}")
