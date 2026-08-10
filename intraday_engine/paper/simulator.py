from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class Position:
    symbol: str
    side: str
    quantity: int
    entry: float
    stop: float
    target: float


class PaperBroker:
    """In-memory paper broker with explicit risk guardrails.

    This class never calls a broker API. It is intentionally suitable for the
    six-month research period before any live execution path is considered.
    """

    def __init__(
        self,
        capital: float,
        risk_per_trade: float = 0.005,
        max_positions: int = 3,
        max_daily_loss: float = 0.02,
        max_position_notional: float = 0.25,
        slippage_points: float = 0.0,
    ):
        if capital <= 0:
            raise ValueError("capital must be > 0")
        if not 0 < risk_per_trade <= 0.05:
            raise ValueError("risk_per_trade must be > 0 and <= 0.05")
        if max_positions < 1:
            raise ValueError("max_positions must be >= 1")
        if not 0 < max_daily_loss <= 0.20:
            raise ValueError("max_daily_loss must be > 0 and <= 0.20")
        if not 0 < max_position_notional <= 1.0:
            raise ValueError("max_position_notional must be > 0 and <= 1")
        if slippage_points < 0:
            raise ValueError("slippage_points must be >= 0")

        self.initial_capital = float(capital)
        self.capital = float(capital)
        self.risk_per_trade = float(risk_per_trade)
        self.max_positions = int(max_positions)
        self.max_daily_loss = float(max_daily_loss)
        self.max_position_notional = float(max_position_notional)
        self.slippage_points = float(slippage_points)
        self.positions: dict[str, Position] = {}
        self.realized_pnl_today = 0.0
        self._day_start_capital = float(capital)
        self._trading_date = None
        self.halted = False

    @staticmethod
    def _date(now: datetime | None) -> object:
        current = now or datetime.now(IST)
        if current.tzinfo is None:
            current = current.replace(tzinfo=IST)
        return current.astimezone(IST).date()

    def _roll_day(self, now: datetime | None = None) -> None:
        trading_date = self._date(now)
        if trading_date != self._trading_date:
            self._trading_date = trading_date
            self._day_start_capital = self.capital
            self.realized_pnl_today = 0.0
            self.halted = False

    def _effective_entry(self, side: str, entry: float) -> float:
        return entry + self.slippage_points if side == "LONG" else entry - self.slippage_points

    def _effective_exit(self, side: str, price: float) -> float:
        return price - self.slippage_points if side == "LONG" else price + self.slippage_points

    def _daily_loss_limit_hit(self) -> bool:
        return self.realized_pnl_today <= -(self._day_start_capital * self.max_daily_loss)

    def size(self, entry: float, stop: float, side: str = "LONG") -> int:
        if entry <= 0 or stop <= 0:
            return 0
        effective_entry = self._effective_entry(side, entry)
        risk = abs(effective_entry - stop)
        if risk <= 0:
            return 0
        risk_budget = self.capital * self.risk_per_trade
        risk_quantity = int(risk_budget / risk)
        notional_quantity = int((self.capital * self.max_position_notional) / effective_entry)
        return max(0, min(risk_quantity, notional_quantity))

    def open(
        self,
        symbol: str,
        side: str,
        entry: float,
        stop: float,
        target: float,
        now: datetime | None = None,
    ) -> Position | None:
        self._roll_day(now)
        side = side.upper()
        if side not in {"LONG", "SHORT"}:
            raise ValueError("side must be LONG or SHORT")
        if entry <= 0 or stop <= 0 or target <= 0:
            return None
        if side == "LONG" and not (stop < entry < target):
            return None
        if side == "SHORT" and not (target < entry < stop):
            return None
        if self.halted or self._daily_loss_limit_hit():
            self.halted = True
            return None
        if len(self.positions) >= self.max_positions or symbol in self.positions:
            return None

        quantity = self.size(entry, stop, side=side)
        if quantity <= 0:
            return None
        effective_entry = self._effective_entry(side, entry)
        position = Position(symbol, side, quantity, effective_entry, float(stop), float(target))
        self.positions[symbol] = position
        return position

    def mark(self, symbol: str, price: float, now: datetime | None = None) -> dict | None:
        self._roll_day(now)
        position = self.positions.get(symbol)
        if not position or price <= 0:
            return None

        hit_stop = price <= position.stop if position.side == "LONG" else price >= position.stop
        hit_target = price >= position.target if position.side == "LONG" else price <= position.target
        if not hit_stop and not hit_target:
            return None

        exit_price = self._effective_exit(position.side, float(price))
        pnl = (
            (exit_price - position.entry) * position.quantity
            if position.side == "LONG"
            else (position.entry - exit_price) * position.quantity
        )
        self.capital += pnl
        self.realized_pnl_today += pnl
        del self.positions[symbol]
        if self._daily_loss_limit_hit():
            self.halted = True
        return {
            "symbol": symbol,
            "side": position.side,
            "quantity": position.quantity,
            "entry_price": position.entry,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": "TARGET" if hit_target and not hit_stop else "STOP",
            "trading_date": self._trading_date,
        }

    def force_close(self, symbol: str, price: float, now: datetime | None = None) -> dict | None:
        """Close an open paper position at the supplied market price."""
        self._roll_day(now)
        position = self.positions.get(symbol)
        if not position or price <= 0:
            return None
        exit_price = self._effective_exit(position.side, float(price))
        pnl = (
            (exit_price - position.entry) * position.quantity
            if position.side == "LONG"
            else (position.entry - exit_price) * position.quantity
        )
        self.capital += pnl
        self.realized_pnl_today += pnl
        del self.positions[symbol]
        if self._daily_loss_limit_hit():
            self.halted = True
        return {
            "symbol": symbol,
            "side": position.side,
            "quantity": position.quantity,
            "entry_price": position.entry,
            "exit_price": exit_price,
            "pnl": pnl,
            "reason": "FORCED_EXIT",
            "trading_date": self._trading_date,
        }

    def status(self, now: datetime | None = None) -> dict:
        self._roll_day(now)
        return {
            "capital": self.capital,
            "realized_pnl_today": self.realized_pnl_today,
            "open_positions": len(self.positions),
            "max_positions": self.max_positions,
            "halted": self.halted,
            "trading_date": self._trading_date,
        }
