"""Research-grade, fully configurable transaction-cost overlay.

This module never runs inside backtest_signals()/evaluate_trade() and never
changes their defaults or behavior -- it is a strictly additive, opt-in
post-hoc layer applied to already-simulated trades (gross pnl_points, as
computed today), so every existing backtest/paper result is completely
unaffected unless a caller explicitly asks for a cost breakdown.

Rate defaults below are standard, widely-cited Indian discount-broker
intraday-equity cash-segment rates (brokerage, STT, NSE exchange transaction
charge, SEBI turnover fee, GST, stamp duty) as commonly quoted in broker cost
calculators. They are illustrative planning assumptions, not values fetched
from a live regulatory rate card -- verify against your actual
broker/exchange notification before treating results as precise. Every rate
is a plain constructor argument; nothing here is hard-coded into strategy
logic, and none of these numbers should ever be tuned to make a strategy
look better -- that would defeat the entire point of separating gross from
net performance.

Brokerage specifically is a flat percentage-of-turnover assumption, not a
true per-order cap (see CostModel's docstring) -- this module has no
order-value/quantity information to evaluate a real "percentage OR flat fee,
whichever is lower" cap against, and does not guess one.

IMPORTANT -- existing paper-trading history: every paper trade recorded so
far (research/paper_outcomes.py::evaluate_trade) was, and still is, recorded
with no cost model applied at all -- its pnl_points/r_multiple are gross,
unadjusted figures. This module does not retroactively touch that stored
history; apply it only when explicitly building a cost-comparison report
over those trades (e.g. via build_cost_comparison_report), never by rewriting
paper_outcomes rows in place.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


@dataclass(frozen=True)
class CostModel:
    """Configurable per-share transaction-cost rates.

    brokerage_pct is a flat percentage-of-turnover approximation, not the
    real "brokerage_pct of turnover OR a flat fee per order, whichever is
    lower" cap most Indian discount brokers actually charge. Implementing
    that real cap correctly requires the order's actual value (price times
    quantity) so the flat-fee side of the comparison means something -- the
    trade representation this module operates on (backtest/engine.py's
    BacktestTrade, research/paper_outcomes.py's outcome dicts) carries prices
    only, never a quantity or notional size, and neither simulator does
    position sizing today. Rather than silently guess a quantity to make a
    "cap" comparison meaningful, brokerage here is deliberately left as an
    uncapped, linear percentage of turnover -- correct as an illustrative
    upper-bound-shaped assumption, but do not read it as "the real discount-
    broker fee schedule."
    """

    entry_slippage_points: float = 0.0
    exit_slippage_points: float = 0.0
    brokerage_pct: float = 0.0003        # 0.03% of turnover per executed leg, flat -- see class docstring: NOT a per-order cap
    stt_sell_pct: float = 0.00025        # 0.025% of the SELL leg's turnover only (intraday equity STT)
    exchange_txn_pct: float = 0.0000297  # ~0.00297% of turnover, both legs (NSE transaction charge)
    sebi_pct: float = 0.000001           # ~Rs 10 per crore of turnover, both legs
    gst_pct: float = 0.18                # 18% GST, levied on (brokerage + exchange charges) only
    stamp_duty_buy_pct: float = 0.00003  # 0.003% of the BUY leg's turnover only

    @classmethod
    def zero_cost(cls) -> "CostModel":
        """No slippage, no charges -- isolates the strategy's raw signal edge."""
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def realistic(cls) -> "CostModel":
        """The documented default rates above, zero slippage (the underlying
        simulator's own slippage_points parameter, if any, already reflects
        whatever execution slippage the caller chose to model)."""
        return cls()

    @classmethod
    def conservative_stress(cls, *, slippage_points: float = 0.05, cost_multiplier: float = 1.5) -> "CostModel":
        """Deliberately pessimistic: adds real per-leg slippage on top of the
        realistic charge rates (inflated by `cost_multiplier`), to check
        whether an apparent edge survives worse-than-typical execution."""
        base = cls.realistic()
        return replace(
            base,
            entry_slippage_points=slippage_points,
            exit_slippage_points=slippage_points,
            brokerage_pct=base.brokerage_pct * cost_multiplier,
            stt_sell_pct=base.stt_sell_pct * cost_multiplier,
            exchange_txn_pct=base.exchange_txn_pct * cost_multiplier,
            sebi_pct=base.sebi_pct * cost_multiplier,
            stamp_duty_buy_pct=base.stamp_duty_buy_pct * cost_multiplier,
        )

    def _leg_cost_pct(self, *, is_buy_leg: bool) -> float:
        brokerage_and_exchange = self.brokerage_pct + self.exchange_txn_pct
        gst = brokerage_and_exchange * self.gst_pct
        stt = 0.0 if is_buy_leg else self.stt_sell_pct
        stamp_duty = self.stamp_duty_buy_pct if is_buy_leg else 0.0
        return brokerage_and_exchange + gst + self.sebi_pct + stt + stamp_duty

    def leg_cost_points(self, price: float, *, is_buy_leg: bool) -> float:
        return price * self._leg_cost_pct(is_buy_leg=is_buy_leg)

    def round_trip_cost_points(self, entry_price: float, exit_price: float, side: str) -> float:
        """Brokerage/STT/exchange/SEBI/GST/stamp-duty for one round trip, in
        price points (percentage-of-turnover costs are independent of
        quantity, so this stays in the same points unit as pnl_points
        regardless of position size)."""
        side = side.upper()
        if side not in {"LONG", "SHORT"}:
            raise ValueError("side must be LONG or SHORT")
        entry_is_buy = side == "LONG"
        return self.leg_cost_points(entry_price, is_buy_leg=entry_is_buy) + self.leg_cost_points(
            exit_price, is_buy_leg=not entry_is_buy
        )

    def round_trip_slippage_points(self) -> float:
        return self.entry_slippage_points + self.exit_slippage_points


@dataclass(frozen=True)
class CostReport:
    trades: int
    gross_pnl_points: float
    slippage_points: float
    transaction_cost_points: float
    net_pnl_points: float
    gross_expectancy_r: float
    net_expectancy_r: float
    gross_profit_factor: float
    net_profit_factor: float


def _profit_factor(pnls: list[float]) -> float:
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    return gross_profit / gross_loss if gross_loss else float("inf")


def apply_cost_model(trades: Iterable[dict], cost_model: CostModel) -> CostReport:
    """Compute gross vs. net performance for already-simulated trades.

    Each trade dict must carry: side ("LONG"/"SHORT"), entry_price,
    exit_price, stop_loss, pnl_points (gross, exactly as the simulator
    already computed it -- this function never re-simulates entries/exits).
    risk (abs(entry_price - stop_loss)) is used to convert points to R,
    matching how backtest/engine.py and paper_outcomes.py already define R.
    """
    rows = list(trades)
    if not rows:
        return CostReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    gross_pnls: list[float] = []
    net_pnls: list[float] = []
    gross_rs: list[float] = []
    net_rs: list[float] = []
    total_slippage = 0.0
    total_txn_cost = 0.0

    round_trip_slippage = cost_model.round_trip_slippage_points()
    for row in rows:
        risk = abs(float(row["entry_price"]) - float(row["stop_loss"]))
        txn_cost = cost_model.round_trip_cost_points(
            float(row["entry_price"]), float(row["exit_price"]), row["side"]
        )
        gross_pnl = float(row["pnl_points"])
        net_pnl = gross_pnl - round_trip_slippage - txn_cost

        gross_pnls.append(gross_pnl)
        net_pnls.append(net_pnl)
        total_slippage += round_trip_slippage
        total_txn_cost += txn_cost
        if risk > 0:
            gross_rs.append(gross_pnl / risk)
            net_rs.append(net_pnl / risk)

    return CostReport(
        trades=len(rows),
        gross_pnl_points=sum(gross_pnls),
        slippage_points=total_slippage,
        transaction_cost_points=total_txn_cost,
        net_pnl_points=sum(net_pnls),
        gross_expectancy_r=(sum(gross_rs) / len(gross_rs)) if gross_rs else 0.0,
        net_expectancy_r=(sum(net_rs) / len(net_rs)) if net_rs else 0.0,
        gross_profit_factor=_profit_factor(gross_pnls),
        net_profit_factor=_profit_factor(net_pnls),
    )


@dataclass(frozen=True)
class CostComparisonReport:
    """The full gross vs. realistic-net vs. conservative-net comparison in
    one structure, so a caller never has to re-derive "gross" three times or
    manually line up three separate CostReport objects by hand."""

    trades: int
    gross_pnl_points: float
    gross_expectancy_r: float
    gross_profit_factor: float
    realistic_net_pnl_points: float
    realistic_net_expectancy_r: float
    realistic_net_profit_factor: float
    conservative_net_pnl_points: float
    conservative_net_expectancy_r: float
    conservative_net_profit_factor: float


def build_cost_comparison_report(
    trades: Iterable[dict],
    *,
    realistic_cost_model: CostModel | None = None,
    conservative_cost_model: CostModel | None = None,
) -> CostComparisonReport:
    """Gross, realistic-net, and conservative-net performance for the same
    trades side by side. Gross is identical under any cost model (it's
    computed before any cost is subtracted) so it's reported once, not
    duplicated per preset.
    """
    rows = list(trades)
    realistic = apply_cost_model(rows, realistic_cost_model or CostModel.realistic())
    conservative = apply_cost_model(rows, conservative_cost_model or CostModel.conservative_stress())

    return CostComparisonReport(
        trades=len(rows),
        gross_pnl_points=realistic.gross_pnl_points,
        gross_expectancy_r=realistic.gross_expectancy_r,
        gross_profit_factor=realistic.gross_profit_factor,
        realistic_net_pnl_points=realistic.net_pnl_points,
        realistic_net_expectancy_r=realistic.net_expectancy_r,
        realistic_net_profit_factor=realistic.net_profit_factor,
        conservative_net_pnl_points=conservative.net_pnl_points,
        conservative_net_expectancy_r=conservative.net_expectancy_r,
        conservative_net_profit_factor=conservative.net_profit_factor,
    )


def backtest_trades_to_cost_rows(trades: Iterable) -> list[dict]:
    """Adapt backtest/engine.py::BacktestTrade instances to the plain-dict
    shape apply_cost_model expects."""
    return [
        {
            "side": trade.side,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "stop_loss": trade.stop_loss,
            "pnl_points": trade.pnl_points,
        }
        for trade in trades
    ]
