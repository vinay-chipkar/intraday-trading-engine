from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    upstox_access_token: str | None = os.getenv("UPSTOX_ACCESS_TOKEN")
    duckdb_path: str = os.getenv("DUCKDB_PATH", str(ROOT / "data" / "trading.duckdb"))
    timezone: str = os.getenv("TIMEZONE", "Asia/Kolkata")
    top_n: int = int(os.getenv("TOP_N", "10"))
    candle_interval: int = int(os.getenv("CANDLE_INTERVAL", "1"))
    paper_initial_capital: float = float(os.getenv("PAPER_INITIAL_CAPITAL", "100000"))
    paper_risk_per_trade: float = float(os.getenv("PAPER_RISK_PER_TRADE", "0.005"))
    paper_max_positions: int = int(os.getenv("PAPER_MAX_POSITIONS", "3"))
    paper_max_daily_loss: float = float(os.getenv("PAPER_MAX_DAILY_LOSS", "0.02"))
    paper_max_position_notional: float = float(os.getenv("PAPER_MAX_POSITION_NOTIONAL", "0.25"))
    paper_slippage_points: float = float(os.getenv("PAPER_SLIPPAGE_POINTS", "0.10"))


settings = Settings()
Path(settings.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
