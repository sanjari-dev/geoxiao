"""Streaming vectorized backtesting via ClickHouse.

This module evaluates one ``StrategyDNA`` by pushing feature calculation,
signal generation, and SL/TP exit discovery into ClickHouse SQL.  Python only
streams the resulting trade rows in small blocks, updates running metrics, and
optionally forwards each block to PostgreSQL trade-log persistence.
"""

from __future__ import annotations

import ast
import math
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

import clickhouse_connect
import structlog

from src.config.settings import settings
from src.strategy.signal_utils import clamp_signal_threshold
from src.strategy.base_strategy import StrategyDNA

log = structlog.get_logger(__name__)


@dataclass
class RunningBacktestMetrics:
    """Incremental metrics updated while streaming ClickHouse result blocks."""

    total_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    winning_trades: int = 0
    cumulative_pips: float = 0.0
    equity_peak: float = 0.0
    max_drawdown_pips: float = 0.0

    def update(self, raw_pips: float | None) -> None:
        if raw_pips is None or math.isnan(float(raw_pips)):
            return

        pips = float(raw_pips)
        self.total_trades += 1
        if pips > 0:
            self.gross_profit += pips
            self.winning_trades += 1
        else:
            self.gross_loss += abs(pips)

        self.cumulative_pips += pips
        self.equity_peak = max(self.equity_peak, self.cumulative_pips)
        drawdown = self.equity_peak - self.cumulative_pips
        self.max_drawdown_pips = max(self.max_drawdown_pips, drawdown)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades else 0.0

    @property
    def profit_factor(self) -> float | None:
        if self.gross_loss == 0:
            return None
        return self.gross_profit / self.gross_loss

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_pips": self.cumulative_pips,
            "max_drawdown_pips": self.max_drawdown_pips,
        }


@dataclass(frozen=True)
class BacktestChunkSpec:
    trade_start: datetime | None
    trade_end: datetime | None
    scan_start: datetime | None
    scan_end: datetime | None


@dataclass(frozen=True)
class UnresolvedEntryInspection:
    timeout_count: int
    first_timeout_entry_time: datetime | None
    pending_latest_count: int = 0
    first_pending_latest_entry_time: datetime | None = None


@dataclass(frozen=True)
class UnresolvedTradeTimeoutError(RuntimeError):
    timeout_count: int
    max_holding_seconds: int
    first_entry_time: datetime | None = None
    partial_metrics: dict[str, Any] | None = None

    def __str__(self) -> str:
        first_entry = (
            f", first_entry_time={self.first_entry_time.isoformat()}"
            if self.first_entry_time is not None
            else ""
        )
        return (
            f"{self.timeout_count} trade(s) exceeded max_holding_seconds="
            f"{self.max_holding_seconds}{first_entry}"
        )


class ClickHouseEvaluator:
    """Evaluate a strategy with ClickHouse SQL and stream trade rows in blocks."""

    FEATURE_NAMES = {
        "obi",
        "tick_vel",
        "spread_dyn",
        "tick_den",
        "vol_skew",
        "mid_mom",
        "skewness",
        "kurtosis",
        "vw_spread",
    }

    BINARY_OPS = {
        "add": "({a} + {b})",
        "sub": "({a} - {b})",
        "mul": "({a} * {b})",
        "div": "if(abs({b}) > 1e-10, {a} / {b}, 1.0)",
        "max2": "greatest({a}, {b})",
        "min2": "least({a}, {b})",
    }

    UNARY_OPS = {
        "neg": "(-{x})",
        "square": "pow({x}, 2)",
        "cube": "pow({x}, 3)",
        "log": "if(abs({x}) > 1e-10, log(abs({x})), 0.0)",
        "sqrt": "sqrt(abs({x}))",
        "sigmoid": "(1.0 / (1.0 + exp(-least(greatest({x}, -500), 500))))",
        "sign": "sign({x})",
    }

    DEFAULT_BLOCK_SIZE = 10_000
    DEFAULT_MAX_HOLDING_SECONDS = 3_600
    DEFAULT_ENTRY_COOLDOWN_SECONDS = 60
    DEFAULT_BACKTEST_CHUNK_DAYS = 7
    DEFAULT_FEATURE_WARMUP_SECONDS = 3_600

    def __init__(
        self,
        *,
        database: str | None = None,
        ticks_table: str = "ticks",
        block_size: int | None = None,
        max_holding_seconds: int | None = None,
        entry_cooldown_seconds: int | None = None,
        chunk_days: int | None = None,
        feature_warmup_seconds: int | None = None,
    ) -> None:
        self.database = database or settings.CH_DATABASE
        self.ticks_table = ticks_table
        self.block_size = block_size or settings.CLICKHOUSE_BLOCK_SIZE
        self.max_holding_seconds = max_holding_seconds or settings.CLICKHOUSE_MAX_HOLDING_SECONDS
        self.entry_cooldown_seconds = entry_cooldown_seconds or settings.CLICKHOUSE_ENTRY_COOLDOWN_SECONDS
        self.chunk_days = chunk_days if chunk_days is not None else settings.CLICKHOUSE_BACKTEST_CHUNK_DAYS
        self.feature_warmup_seconds = (
            feature_warmup_seconds
            if feature_warmup_seconds is not None
            else settings.CLICKHOUSE_FEATURE_WARMUP_SECONDS
        )
        self.client = clickhouse_connect.get_client(
            host=settings.CH_HOST,
            port=settings.CH_PORT,
            database=self.database,
            username=settings.CH_USER,
            password=settings.CH_PASSWORD,
            connect_timeout=10,
            send_receive_timeout=600,
        )

    def _translate_dna_to_sql(self, dna: StrategyDNA) -> str:
        """Translate a DEAP GP tree expression into a ClickHouse SQL expression."""

        try:
            parsed = ast.parse(dna.tree_repr, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid DNA tree expression: {dna.tree_repr}") from exc

        return self._translate_ast_node(parsed.body)

    def evaluate_stream(
        self,
        dna: StrategyDNA,
        *,
        trial_id: str | None = None,
        persist_trade_sink: Any | None = None,
    ) -> dict[str, Any]:
        """Run a ClickHouse backtest query and stream the trade logs.

        Args:
            dna: Strategy DNA to evaluate.
            trial_id: Optional PostgreSQL trial id used when formatting trade logs.
            persist_trade_sink: Optional sink for persistence.  Supported shapes:
                an ``AsyncTradeLogSync``-like object with ``enqueue(trade)`` or a
                callable that accepts ``list[dict]`` per streamed block.

        Returns:
            Incremental aggregate metrics computed while streaming rows.
        """

        metrics = RunningBacktestMetrics()
        trade_id = trial_id or str(uuid.uuid4())

        log.info(
            "ClickHouse streaming backtest starting",
            strategy_id=dna.id,
            symbol=dna.symbol,
            block_size=self.block_size,
            chunk_days=self.chunk_days,
            max_holding_seconds=self.max_holding_seconds,
        )

        for chunk in self._iter_backtest_chunks(dna):
            unresolved = self._inspect_unresolved_entries(
                dna,
                chunk=chunk,
            )
            if unresolved.pending_latest_count > 0:
                log.info(
                    "Allowing unresolved latest-edge entries due to incomplete future data",
                    strategy_id=dna.id,
                    pending_latest_count=unresolved.pending_latest_count,
                    first_pending_latest_entry_time=(
                        unresolved.first_pending_latest_entry_time.isoformat()
                        if unresolved.first_pending_latest_entry_time
                        else None
                    ),
                )
            if unresolved.timeout_count > 0:
                raise UnresolvedTradeTimeoutError(
                    timeout_count=unresolved.timeout_count,
                    max_holding_seconds=self.max_holding_seconds,
                    first_entry_time=unresolved.first_timeout_entry_time,
                    partial_metrics=metrics.as_dict(),
                )

            query = self._build_query(
                dna,
                trade_start=chunk.trade_start,
                trade_end=chunk.trade_end,
                scan_start=chunk.scan_start,
                scan_end=chunk.scan_end,
            )
            self._stream_query(
                query,
                dna=dna,
                trial_id=trade_id,
                metrics=metrics,
                persist_trade_sink=persist_trade_sink,
            )

        result = metrics.as_dict()
        result["trial_id"] = trade_id
        log.info("ClickHouse streaming backtest complete", strategy_id=dna.id, **result)
        return result

    def _stream_query(
        self,
        query: str,
        *,
        dna: StrategyDNA,
        trial_id: str,
        metrics: RunningBacktestMetrics,
        persist_trade_sink: Any | None,
    ) -> None:
        with self.client.query_row_block_stream(
            query,
            settings={
                "max_block_size": self.block_size,
                "max_threads": settings.CLICKHOUSE_QUERY_MAX_THREADS,
                # Required by some ClickHouse versions for JOIN predicates that
                # include a timestamp range in addition to the equality key.
                "allow_experimental_join_condition": 1,
            },
        ) as stream:
            for block in stream:
                batch = self._process_block(block, dna=dna, trial_id=trial_id, metrics=metrics)
                if persist_trade_sink is not None and batch:
                    self._flush_batch_to_sink(persist_trade_sink, batch)
                del batch

    def _iter_backtest_chunks(self, dna: StrategyDNA) -> Iterable[BacktestChunkSpec]:
        if self.chunk_days <= 0:
            yield BacktestChunkSpec(
                trade_start=None,
                trade_end=None,
                scan_start=None,
                scan_end=None,
            )
            return

        start, end = self._fetch_symbol_bounds(dna.symbol)
        if start is None or end is None or start >= end:
            return

        chunk_delta = timedelta(days=self.chunk_days)
        holding_delta = timedelta(seconds=self.max_holding_seconds)
        feature_windows = self._feature_window_config(dna.params or {})
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + chunk_delta, end + timedelta(microseconds=1))
            scan_start = self._resolve_scan_start(
                dna.symbol,
                chunk_start=chunk_start,
                absolute_start=start,
                feature_windows=feature_windows,
            )
            scan_end = chunk_end + holding_delta
            log.info(
                "ClickHouse backtest chunk",
                strategy_id=dna.id,
                symbol=dna.symbol,
                trade_start=chunk_start.isoformat(),
                trade_end=chunk_end.isoformat(),
            )
            yield BacktestChunkSpec(
                trade_start=chunk_start,
                trade_end=chunk_end,
                scan_start=scan_start,
                scan_end=scan_end,
            )
            chunk_start = chunk_end

    def _fetch_symbol_bounds(self, symbol: str) -> tuple[datetime | None, datetime | None]:
        query = f"""
SELECT
    min(timestamp) AS start_time,
    max(timestamp) AS end_time
FROM {self._quoted_table()}
WHERE instrument = '{self._escape_literal(symbol)}'
"""
        row = self.client.query(query).first_row
        if row is None or row[0] is None or row[1] is None:
            return None, None
        return self._ensure_datetime(row[0]), self._ensure_datetime(row[1])

    def _build_query(
        self,
        dna: StrategyDNA,
        *,
        trade_start: datetime | None = None,
        trade_end: datetime | None = None,
        scan_start: datetime | None = None,
        scan_end: datetime | None = None,
    ) -> str:
        cte_sql = self._build_backtest_cte_sql(
            dna,
            trade_start=trade_start,
            trade_end=trade_end,
            scan_start=scan_start,
            scan_end=scan_end,
        )

        return f"""
{cte_sql},
resolved_exits AS (
    SELECT
        h.entry_time,
        h.exit_time,
        h.side,
        h.entry_price,
        multiIf(
            h.side = 'BUY' AND t.bid <= h.sl_price, h.sl_price,
            h.side = 'BUY' AND t.bid >= h.tp_price, h.tp_price,
            h.side = 'SELL' AND t.ask >= h.sl_price, h.sl_price,
            h.side = 'SELL' AND t.ask <= h.tp_price, h.tp_price,
            h.entry_price
        ) AS exit_price
    FROM exit_hits AS h
    INNER JOIN base_ticks AS t
        ON t.instrument = '{self._escape_literal(dna.symbol)}'
       AND t.timestamp = h.exit_time
)
SELECT
    entry_time,
    exit_time,
    side,
    entry_price,
    exit_price,
    if(side = 'BUY',
       (exit_price - entry_price) / pip,
       (entry_price - exit_price) / pip) AS raw_pips
FROM resolved_exits
ORDER BY entry_time ASC
"""

    def _build_timeout_count_query(
        self,
        dna: StrategyDNA,
        *,
        trade_start: datetime | None = None,
        trade_end: datetime | None = None,
        scan_start: datetime | None = None,
        scan_end: datetime | None = None,
    ) -> str:
        cte_sql = self._build_backtest_cte_sql(
            dna,
            trade_start=trade_start,
            trade_end=trade_end,
            scan_start=scan_start,
            scan_end=scan_end,
        )

        return f"""
{cte_sql}
SELECT
    countIf(
        e.entry_time + toIntervalSecond(max_holding_seconds) <= data_bounds.last_tick
    ) AS timeout_count,
    minIf(
        e.entry_time,
        e.entry_time + toIntervalSecond(max_holding_seconds) <= data_bounds.last_tick
    ) AS first_timeout_entry_time,
    countIf(
        e.entry_time + toIntervalSecond(max_holding_seconds) > data_bounds.last_tick
    ) AS pending_latest_count,
    minIf(
        e.entry_time,
        e.entry_time + toIntervalSecond(max_holding_seconds) > data_bounds.last_tick
    ) AS first_pending_latest_entry_time
FROM eligible_entries AS e
LEFT JOIN exit_hits AS h
    ON h.entry_time = e.entry_time
   AND h.side = e.side
   AND h.entry_price = e.entry_price
   AND h.sl_price = e.sl_price
   AND h.tp_price = e.tp_price
CROSS JOIN (
    SELECT max(timestamp) AS last_tick
    FROM base_ticks
) AS data_bounds
WHERE h.exit_time IS NULL
"""

    def _build_backtest_cte_sql(
        self,
        dna: StrategyDNA,
        *,
        trade_start: datetime | None = None,
        trade_end: datetime | None = None,
        scan_start: datetime | None = None,
        scan_end: datetime | None = None,
    ) -> str:
        signal_sql = self._translate_dna_to_sql(dna)
        params = dna.params or {}
        sl_pips = float(params.get("sl_pips", 20.0))
        tp_pips = float(params.get("tp_pips", 40.0))
        threshold = clamp_signal_threshold(params.get("signal_threshold", 0.5))
        normalized_signal_sql = self._normalize_signal_sql(signal_sql)

        feature_windows = self._feature_window_config(params)
        tick_filters = [f"instrument = '{self._escape_literal(dna.symbol)}'"]
        if scan_start is not None:
            tick_filters.append(f"timestamp >= {self._datetime64_literal(scan_start)}")
        if scan_end is not None:
            tick_filters.append(f"timestamp < {self._datetime64_literal(scan_end)}")
        tick_where = "\n      AND ".join(tick_filters)
        entry_filters: list[str] = []
        if trade_start is not None:
            entry_filters.append(f"entry_time >= {self._datetime64_literal(trade_start)}")
        if trade_end is not None:
            entry_filters.append(f"entry_time < {self._datetime64_literal(trade_end)}")
        entry_where = (
            "WHERE " + "\n      AND ".join(entry_filters)
            if entry_filters
            else ""
        )

        return f"""
WITH
    {sl_pips:.8f} AS sl_pips,
    {tp_pips:.8f} AS tp_pips,
    {threshold:.8f} AS signal_threshold,
    {self.max_holding_seconds:d} AS max_holding_seconds,
    {self.entry_cooldown_seconds:d} AS entry_cooldown_seconds,
    0.0001 AS pip,
base_ticks AS (
    SELECT
        timestamp,
        instrument,
        toFloat64(bid) AS bid,
        toFloat64(ask) AS ask,
        toFloat64(bid_volume) AS bid_size,
        toFloat64(ask_volume) AS ask_size,
	        (toFloat64(bid) + toFloat64(ask)) / 2.0 AS mid,
	        toFloat64(ask) - toFloat64(bid) AS spread
	    FROM {self._quoted_table()}
	    WHERE {tick_where}
	),
ordered_ticks AS (
    SELECT
        *,
        lagInFrame(mid, 1, mid) OVER (
            PARTITION BY instrument ORDER BY timestamp
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS prev_mid,
        mid - prev_mid AS mid_delta
    FROM base_ticks
),
features AS (
    SELECT
        *,
        if(
            sum(bid_size + ask_size) OVER w_obi > 0,
            (sum(bid_size) OVER w_obi - sum(ask_size) OVER w_obi)
                / sum(bid_size + ask_size) OVER w_obi,
            0.0
        ) AS obi,
        avg(abs(mid_delta)) OVER w_tick_vel * 10000 AS tick_vel,
        if(avg(spread) OVER w_spread_dyn > 1e-10,
           stddevPop(spread) OVER w_spread_dyn / avg(spread) OVER w_spread_dyn,
           0.0) AS spread_dyn,
        count() OVER w_tick_den / {feature_windows["tick_den"]:.6f} AS tick_den,
        if(
            stddevPop(bid_size) OVER w_vol_skew + stddevPop(ask_size) OVER w_vol_skew > 1e-10,
            (stddevPop(bid_size) OVER w_vol_skew - stddevPop(ask_size) OVER w_vol_skew)
                / (stddevPop(bid_size) OVER w_vol_skew + stddevPop(ask_size) OVER w_vol_skew),
            0.0
        ) AS vol_skew,
        (avg(mid) OVER w_mid_short - avg(mid) OVER w_mid_long) * 10000 AS mid_mom,
        skewPop(mid_delta) OVER w_skew AS skewness,
        kurtPop(mid_delta) OVER w_kurt AS kurtosis,
        if(
            sum(bid_size + ask_size) OVER w_vw_spread > 1e-10,
            sum((spread * 10000) * (bid_size + ask_size)) OVER w_vw_spread
                / sum(bid_size + ask_size) OVER w_vw_spread,
            0.0
        ) AS vw_spread
    FROM ordered_ticks
    WINDOW
        w_obi AS (
            PARTITION BY instrument ORDER BY timestamp
            ROWS BETWEEN {feature_windows["obi"] - 1:d} PRECEDING AND CURRENT ROW
        ),
        w_tick_vel AS (
            PARTITION BY instrument ORDER BY timestamp
            ROWS BETWEEN {feature_windows["tick_vel"] - 1:d} PRECEDING AND CURRENT ROW
        ),
        w_spread_dyn AS (
            PARTITION BY instrument ORDER BY timestamp
            ROWS BETWEEN {feature_windows["spread_dyn"] - 1:d} PRECEDING AND CURRENT ROW
        ),
        w_tick_den AS (
            PARTITION BY instrument ORDER BY timestamp
            ROWS BETWEEN {feature_windows["tick_den_rows"] - 1:d} PRECEDING AND CURRENT ROW
        ),
        w_vol_skew AS (
            PARTITION BY instrument ORDER BY timestamp
            ROWS BETWEEN {feature_windows["vol_skew"] - 1:d} PRECEDING AND CURRENT ROW
        ),
        w_mid_short AS (
            PARTITION BY instrument ORDER BY timestamp
            ROWS BETWEEN {feature_windows["mid_mom_short"] - 1:d} PRECEDING AND CURRENT ROW
        ),
        w_mid_long AS (
            PARTITION BY instrument ORDER BY timestamp
            ROWS BETWEEN {feature_windows["mid_mom_long"] - 1:d} PRECEDING AND CURRENT ROW
        ),
        w_skew AS (
            PARTITION BY instrument ORDER BY timestamp
            ROWS BETWEEN {feature_windows["skew"] - 1:d} PRECEDING AND CURRENT ROW
        ),
        w_kurt AS (
            PARTITION BY instrument ORDER BY timestamp
            ROWS BETWEEN {feature_windows["kurt"] - 1:d} PRECEDING AND CURRENT ROW
        ),
        w_vw_spread AS (
            PARTITION BY instrument ORDER BY timestamp
            ROWS BETWEEN {feature_windows["vw_spread"] - 1:d} PRECEDING AND CURRENT ROW
        )
),
signal_base AS (
    SELECT
        *,
        {normalized_signal_sql} AS signal_value,
        multiIf(signal_value > signal_threshold, 'BUY',
                signal_value < -signal_threshold, 'SELL',
                'HOLD') AS side
    FROM features
),
signals AS (
    SELECT
        *,
        lagInFrame(side, 1, 'HOLD') OVER (
            PARTITION BY instrument ORDER BY timestamp
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS prev_side
    FROM signal_base
),
entries AS (
    SELECT
        timestamp AS entry_time,
        instrument,
        side,
        if(side = 'BUY', ask, bid) AS entry_price,
        if(side = 'BUY', ask - (sl_pips * pip), bid + (sl_pips * pip)) AS sl_price,
        if(side = 'BUY', ask + (tp_pips * pip), bid - (tp_pips * pip)) AS tp_price
    FROM signals
    WHERE side IN ('BUY', 'SELL')
      AND side != prev_side
),
entry_candidates AS (
    SELECT
        *,
        lagInFrame(entry_time, 1, toDateTime64('1970-01-01 00:00:00', 6, 'UTC')) OVER (
            PARTITION BY instrument ORDER BY entry_time
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS previous_entry_time
    FROM entries
),
	filtered_entries AS (
	    SELECT *
	    FROM entry_candidates
	    WHERE dateDiff('second', previous_entry_time, entry_time) >= entry_cooldown_seconds
	),
	eligible_entries AS (
	    SELECT *
	    FROM filtered_entries
	    {entry_where}
	),
exit_hits AS (
    SELECT
        e.entry_time,
        e.side,
        e.entry_price,
        e.sl_price,
        e.tp_price,
        minIf(
            f.timestamp,
            (e.side = 'BUY' AND (f.bid <= e.sl_price OR f.bid >= e.tp_price))
             OR (e.side = 'SELL' AND (f.ask >= e.sl_price OR f.ask <= e.tp_price))
        ) AS exit_time
	    FROM eligible_entries AS e
    INNER JOIN base_ticks AS f
        ON f.instrument = e.instrument
       AND f.timestamp > e.entry_time
       AND f.timestamp <= e.entry_time + toIntervalSecond(max_holding_seconds)
    GROUP BY
        e.entry_time,
        e.side,
        e.entry_price,
        e.sl_price,
        e.tp_price
    HAVING exit_time > toDateTime64('1970-01-01 00:00:00', 6, 'UTC')
"""

    def _inspect_unresolved_entries(
        self,
        dna: StrategyDNA,
        *,
        chunk: BacktestChunkSpec,
    ) -> UnresolvedEntryInspection:
        query = self._build_timeout_count_query(
            dna,
            trade_start=chunk.trade_start,
            trade_end=chunk.trade_end,
            scan_start=chunk.scan_start,
            scan_end=chunk.scan_end,
        )
        row = self.client.query(query).first_row
        if row is None:
            return UnresolvedEntryInspection(0, None, 0, None)
        return UnresolvedEntryInspection(
            timeout_count=int(row[0] or 0),
            first_timeout_entry_time=(
                self._ensure_datetime(row[1]) if row[1] is not None else None
            ),
            pending_latest_count=int(row[2] or 0),
            first_pending_latest_entry_time=(
                self._ensure_datetime(row[3]) if row[3] is not None else None
            ),
        )

    def _process_block(
        self,
        block: Iterable[tuple[Any, ...]],
        *,
        dna: StrategyDNA,
        trial_id: str,
        metrics: RunningBacktestMetrics,
    ) -> list[dict[str, Any]]:
        batch: list[dict[str, Any]] = []
        for row in block:
            entry_time, exit_time, side, entry_price, exit_price, raw_pips = row
            pips = float(raw_pips) if raw_pips is not None else None
            metrics.update(pips)
            batch.append(
                {
                    "trial_id": trial_id,
                    "strategy_id": dna.id,
                    "symbol": dna.symbol,
                    "side": str(side),
                    "order_type": "LIMIT",
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "sl_price": self._sl_price(str(side), float(entry_price), dna),
                    "tp_price": self._tp_price(str(side), float(entry_price), dna),
                    "entry_time": self._ensure_datetime(entry_time),
                    "exit_time": self._ensure_datetime(exit_time),
                    "raw_pips": pips,
                    "spread_pips": 0.0,
                    "slippage_pips": 0.0,
                    "commission_pips": 0.0,
                    "exit_reason": "SQL_SL_TP",
                    "backtest_month": self._month_start(entry_time),
                }
            )
        return batch

    def _translate_ast_node(self, node: ast.AST) -> str:
        if isinstance(node, ast.Expression):
            return self._translate_ast_node(node.body)

        if isinstance(node, ast.Name):
            if node.id not in self.FEATURE_NAMES:
                raise ValueError(f"Unsupported feature or identifier in DNA tree: {node.id}")
            return node.id

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"Unsupported constant in DNA tree: {node.value!r}")
            return repr(float(node.value))

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"(-{self._translate_ast_node(node.operand)})"

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn_name = node.func.id
            args = [self._translate_ast_node(arg) for arg in node.args]
            if fn_name in self.BINARY_OPS:
                if len(args) != 2:
                    raise ValueError(f"{fn_name} expects 2 arguments")
                return self.BINARY_OPS[fn_name].format(a=args[0], b=args[1])
            if fn_name in self.UNARY_OPS:
                if len(args) != 1:
                    raise ValueError(f"{fn_name} expects 1 argument")
                return self.UNARY_OPS[fn_name].format(x=args[0])
            raise ValueError(f"Unsupported GP primitive in DNA tree: {fn_name}")

        raise ValueError(f"Unsupported DNA AST node: {ast.dump(node)}")

    @staticmethod
    def _normalize_signal_sql(signal_sql: str) -> str:
        return f"tanh(greatest(least(({signal_sql}), 20.0), -20.0))"

    @staticmethod
    def _feature_window_config(params: dict[str, Any]) -> dict[str, int | float]:
        return {
            "obi": int(params.get("obi_window", 20)),
            "tick_vel": int(params.get("tick_vel_window", 10)),
            "spread_dyn": int(params.get("spread_dyn_window", 20)),
            "tick_den": float(params.get("tick_den_window_sec", 60.0)),
            "tick_den_rows": int(params.get("tick_den_rows", 200)),
            "vol_skew": int(params.get("vol_skew_window", 30)),
            "mid_mom_short": int(params.get("mid_mom_short", 5)),
            "mid_mom_long": int(params.get("mid_mom_long", 20)),
            "skew": int(params.get("skew_window", 30)),
            "kurt": int(params.get("kurt_window", 30)),
            "vw_spread": int(params.get("vw_spread_window", 20)),
        }

    @staticmethod
    def _required_feature_warmup_rows(feature_windows: dict[str, int | float]) -> int:
        return max(
            int(feature_windows["obi"]),
            int(feature_windows["tick_vel"]),
            int(feature_windows["spread_dyn"]),
            int(feature_windows["tick_den_rows"]),
            int(feature_windows["vol_skew"]),
            int(feature_windows["mid_mom_long"]),
            int(feature_windows["skew"]),
            int(feature_windows["kurt"]),
            int(feature_windows["vw_spread"]),
            2,
        ) + 1

    def _resolve_scan_start(
        self,
        symbol: str,
        *,
        chunk_start: datetime,
        absolute_start: datetime,
        feature_windows: dict[str, int | float],
    ) -> datetime:
        """
        Resolve scan_start using required historical row lookback, with a
        conservative time-based fallback to preserve cooldown continuity.
        """

        fallback_start = chunk_start - timedelta(
            seconds=max(self.feature_warmup_seconds, self.entry_cooldown_seconds)
        )
        rows_needed = self._required_feature_warmup_rows(feature_windows)
        row_based_start = self._fetch_row_warmup_start(symbol, chunk_start, rows_needed)
        if row_based_start is None:
            return max(absolute_start, fallback_start)
        return max(absolute_start, min(fallback_start, row_based_start))

    def _fetch_row_warmup_start(
        self,
        symbol: str,
        chunk_start: datetime,
        rows_needed: int,
    ) -> datetime | None:
        if rows_needed <= 0:
            return None

        query = f"""
SELECT min(timestamp) AS scan_start
FROM (
    SELECT timestamp
    FROM {self._quoted_table()}
    WHERE instrument = '{self._escape_literal(symbol)}'
      AND timestamp < {self._datetime64_literal(chunk_start)}
    ORDER BY timestamp DESC
    LIMIT {rows_needed:d}
)
"""
        row = self.client.query(query).first_row
        if row is None or row[0] is None:
            return None
        return self._ensure_datetime(row[0])

    def _quoted_table(self) -> str:
        return f"{self._quote_identifier(self.database)}.{self._quote_identifier(self.ticks_table)}"

    @staticmethod
    def _quote_identifier(value: str) -> str:
        return f"`{value.replace('`', '``')}`"

    @staticmethod
    def _escape_literal(value: str) -> str:
        return value.replace("\\", "\\\\").replace("'", "\\'")

    @classmethod
    def _datetime64_literal(cls, value: datetime) -> str:
        dt = cls._ensure_datetime(value).astimezone(timezone.utc)
        return f"toDateTime64('{dt:%Y-%m-%d %H:%M:%S.%f}', 6, 'UTC')"

    @staticmethod
    def _ensure_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))

    @classmethod
    def _month_start(cls, value: Any) -> date:
        dt = cls._ensure_datetime(value)
        return date(dt.year, dt.month, 1)

    @staticmethod
    def _sl_price(side: str, entry_price: float, dna: StrategyDNA) -> float:
        sl_pips = float((dna.params or {}).get("sl_pips", 20.0))
        return entry_price - sl_pips * 0.0001 if side == "BUY" else entry_price + sl_pips * 0.0001

    @staticmethod
    def _tp_price(side: str, entry_price: float, dna: StrategyDNA) -> float:
        tp_pips = float((dna.params or {}).get("tp_pips", 40.0))
        return entry_price + tp_pips * 0.0001 if side == "BUY" else entry_price - tp_pips * 0.0001

    @staticmethod
    def _flush_batch_to_sink(sink: Any, batch: list[dict[str, Any]]) -> None:
        enqueue = getattr(sink, "enqueue", None)
        if callable(enqueue):
            for trade in batch:
                enqueue(trade)
            return
        if callable(sink):
            sink(batch)
            return
        raise TypeError("persist_trade_sink must be callable or expose enqueue(trade)")


__all__ = [
    "BacktestChunkSpec",
    "ClickHouseEvaluator",
    "RunningBacktestMetrics",
    "UnresolvedEntryInspection",
    "UnresolvedTradeTimeoutError",
]
