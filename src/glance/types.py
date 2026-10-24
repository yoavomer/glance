from dataclasses import dataclass, field


@dataclass
class Element:
    id: str  # "e3"; unique within one observation only
    role: str  # "button", "link", "textbox", ...
    label: str  # visible text or accessible name, kept short
    x: int  # center point, screenshot/viewport pixels
    y: int

    def render(self) -> str:
        return f'[{self.id}] {self.role} "{self.label}" @ ({self.x}, {self.y})'


@dataclass
class Observation:
    """Everything the orchestrator learns about the screen. Text only."""

    summary: str
    elements: list[Element]
    answer: str
    url: str = ""  # from the environment, not from vision
    title: str = ""
    usage: dict = field(default_factory=dict)  # perception-call accounting; never rendered

    def render(self) -> str:
        lines = []
        if self.url:
            lines.append(f"Page: {self.title} <{self.url}>")
        lines.append(f"Screen: {self.summary}")
        lines.append(f"Answer: {self.answer}")
        if self.elements:
            lines.append("Elements:")
            lines.extend("  " + e.render() for e in self.elements)
        return "\n".join(lines)
