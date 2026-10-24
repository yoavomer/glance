"""Milestone 1: does the browser Environment do what it claims?

Part 1 uses a local page with a button at a known position, so we can check
that click(x, y) lands exactly where the screenshot says it should.
Part 2 drives real Wikipedia. Playwright's DOM queries are used here ONLY to
find coordinates for the test; the agent itself never gets DOM access.

    python scripts/smoke_env.py [--headed]
"""

import struct
import sys
from pathlib import Path

from glance.env import PlaywrightBrowser

OUT = Path("runs/smoke_env")

TEST_PAGE = """data:text/html,
<body style="margin:0">
  <button id="b" style="position:absolute; left:600px; top:380px; width:80px; height:40px"
          onclick="this.textContent='clicked'">click me</button>
</body>"""


def png_size(png: bytes) -> tuple[int, int]:
    return struct.unpack(">II", png[16:24])


def main(headless: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with PlaywrightBrowser(headless=headless) as env:
        # Part 1: coordinate sanity on a page we control.
        env.navigate(TEST_PAGE)
        shot = env.screenshot()
        assert png_size(shot) == (env.width, env.height), png_size(shot)
        env.click(5, 5)  # miss: empty space
        assert env.page.inner_text("#b") == "click me"
        env.click(640, 400)  # button centre
        assert env.page.inner_text("#b") == "clicked"
        print(f"[ok] screenshot is {env.width}x{env.height}; click(640, 400) hit the button, click(5, 5) did not")

        # Part 2: a real site.
        env.navigate("https://en.wikipedia.org")
        (OUT / "1_home.png").write_bytes(env.screenshot())
        box = env.page.locator("input[name=search]").first.bounding_box()
        x, y = int(box["x"] + box["width"] / 2), int(box["y"] + box["height"] / 2)
        env.click(x, y)
        env.type("perception")
        env.key("Enter")
        (OUT / "2_after_search.png").write_bytes(env.screenshot())
        assert "Main_Page" not in env.metadata()["url"], "search did not navigate (settle returned too early?)"
        print(f"[ok] clicked search box at ({x}, {y}), typed, pressed Enter -> {env.metadata()}")

        env.scroll(0, 600)
        (OUT / "3_scrolled.png").write_bytes(env.screenshot())
        scroll_y = env.page.evaluate("window.scrollY")
        assert scroll_y > 0, scroll_y
        print(f"[ok] scroll(0, 600) -> window.scrollY = {scroll_y}")

    print(f"screenshots in {OUT}/")


if __name__ == "__main__":
    main(headless="--headed" not in sys.argv)
