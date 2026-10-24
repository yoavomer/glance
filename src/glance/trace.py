"""Per-run trace: one JSON line per step, plus the screenshots perception saw.

Screenshots are written for humans debugging a run. Nothing reads them back
into the agent.
"""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path


class Trace:
    def __init__(self, root: Path = Path("runs")):
        self.dir = root / datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._file = (self.dir / "trace.jsonl").open("a")

    def screenshot(self, step: int, png: bytes) -> None:
        (self.dir / f"step{step:02d}.png").write_bytes(png)

    def write(self, record: dict) -> None:
        record = {k: asdict(v) if is_dataclass(v) else v for k, v in record.items()}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
