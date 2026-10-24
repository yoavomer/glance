"""Playwright-backed Environment.

The viewport is fixed and device_scale_factor is 1, so screenshot pixels,
CSS pixels and mouse coordinates are the same space. No coordinate
conversion happens anywhere in this module.
"""

import time

from playwright.sync_api import Page, Request, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeout

from glance.env.base import EnvironmentClosed

NAV_START_GRACE_MS = 300  # time for a click/keypress to kick off a navigation
NAV_COMMIT_TIMEOUT_S = 10.0
IDLE_TIMEOUT_MS = 3000


class PlaywrightBrowser:
    def __init__(self, width: int = 1280, height: int = 800, headless: bool = True):
        self.width = width
        self.height = height
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self._context = self._browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=1,
        )
        # Links with target=_blank open a new page; follow it so the agent
        # keeps looking at whatever the user would be looking at.
        self._context.on("page", self._on_new_page)
        self._nav_pending = False
        self.page: Page = self._context.new_page()

    def _on_new_page(self, page: Page) -> None:
        self.page = page
        page.on("request", self._on_request)
        page.on("requestfailed", self._on_request_failed)
        page.on("framenavigated", self._on_frame_navigated)
        page.on("close", self._on_page_close)

    def _on_page_close(self, page: Page) -> None:
        # If the tab we were driving closes, carry on in the most recent one still open.
        if page is self.page:
            open_pages = [p for p in self._context.pages if not p.is_closed()]
            if open_pages:
                self.page = open_pages[-1]

    def _live_page(self) -> Page:
        if self.page.is_closed() or not self._browser.is_connected():
            raise EnvironmentClosed("the browser window was closed")
        return self.page

    def _on_request(self, req: Request) -> None:
        if req.is_navigation_request() and req.frame == self.page.main_frame:
            self._nav_pending = True

    def _on_request_failed(self, req: Request) -> None:
        if req.is_navigation_request() and req.frame == self.page.main_frame:
            self._nav_pending = False

    def _on_frame_navigated(self, frame) -> None:
        if frame == self.page.main_frame:
            self._nav_pending = False

    def _settle(self) -> None:
        """Wait for the page to stop changing after an action.

        networkidle alone is not enough: right after pressing Enter the
        network is still idle because the navigation hasn't started, and
        wait_for_load_state reports on the *old* document until the new one
        commits. So: give the navigation a moment to start, wait for it to
        commit if it did, then wait for load and (best-effort) network idle.
        """
        self.page.wait_for_timeout(NAV_START_GRACE_MS)
        deadline = time.monotonic() + NAV_COMMIT_TIMEOUT_S
        while self._nav_pending and time.monotonic() < deadline:
            self.page.wait_for_timeout(50)
        try:
            self.page.wait_for_load_state("load", timeout=IDLE_TIMEOUT_MS)
            self.page.wait_for_load_state("networkidle", timeout=IDLE_TIMEOUT_MS)
        except PlaywrightTimeout:
            pass  # Pages with long-polling never go idle; take the screenshot anyway.

    def _check_bounds(self, x: int, y: int) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"({x}, {y}) is outside the {self.width}x{self.height} viewport")

    def screenshot(self) -> bytes:
        return self._live_page().screenshot(type="png")

    def click(self, x: int, y: int) -> None:
        self._check_bounds(x, y)
        self.page.mouse.click(x, y)
        self._settle()

    def type(self, text: str) -> None:
        self.page.keyboard.type(text)
        self._settle()

    def key(self, name: str) -> None:
        self.page.keyboard.press(name)
        self._settle()

    def scroll(self, dx: int, dy: int) -> None:
        # Wheel events go to the element under the pointer, so aim at the middle.
        self.page.mouse.move(self.width // 2, self.height // 2)
        self.page.mouse.wheel(dx, dy)
        self._settle()

    def navigate(self, url: str) -> None:
        if not url.startswith(("http://", "https://", "file:", "data:", "about:")):
            url = "https://" + url
        self.page.goto(url, wait_until="domcontentloaded")
        self._settle()

    def metadata(self) -> dict[str, str]:
        page = self._live_page()
        return {"url": page.url, "title": page.title()}

    def close(self) -> None:
        if self._browser.is_connected():
            self._context.close()
            self._browser.close()
        self._pw.stop()

    def __enter__(self) -> "PlaywrightBrowser":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
