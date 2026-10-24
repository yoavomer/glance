"""The action surface the agent loop drives.

Kept deliberately small so a desktop backend (e.g. pyautogui + mss) can
implement the same Protocol later without touching the loop.
"""

from typing import Protocol


class EnvironmentClosed(RuntimeError):
    """The thing being controlled is gone (e.g. the user closed the browser window)."""


class Environment(Protocol):
    width: int
    height: int

    def screenshot(self) -> bytes:
        """PNG of the visible area. Pixel (x, y) here is the same point click(x, y) hits."""
        ...

    def click(self, x: int, y: int) -> None: ...

    def type(self, text: str) -> None: ...

    def key(self, name: str) -> None:
        """A key or chord, e.g. "Enter", "Tab", "Control+A"."""
        ...

    def scroll(self, dx: int, dy: int) -> None:
        """Scroll by (dx, dy) pixels. Positive dy scrolls down."""
        ...

    def navigate(self, url: str) -> None: ...

    def metadata(self) -> dict[str, str]:
        """Cheap text facts about the current state (e.g. url, title). May be empty."""
        ...

    def close(self) -> None: ...
