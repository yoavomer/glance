import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root regardless of the caller's working directory.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@dataclass(frozen=True)
class Settings:
    orchestrator_model: str = os.getenv("ORCHESTRATOR_MODEL", "claude-opus-5-5")
    perception_model: str = os.getenv("PERCEPTION_MODEL", "claude-haiku-5-5")


settings = Settings()
