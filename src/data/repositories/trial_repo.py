"""Repository for ``trial_logs`` records."""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog

from src.data.repositories.base import (
    AsyncPostgresRepository,
    as_decimal,
    as_int,
    as_json,
    as_uuid,
    from_json,
    require_keys,
)

log = structlog.get_logger(__name__)


class TrialRepository(AsyncPostgresRepository):
    """Persist Optuna/direct-backtest trial summaries to PostgreSQL."""

    _UPSERT_SQL = """
        INSERT INTO trial_logs (
            id, strategy_id, optuna_trial_id, study_name, params_json,
            profit_factor, total_pips, max_drawdown_pips, trade_count,
            fitness_score, eliminated_reason, duration_sec
        )
        VALUES ($1,$2,$3,$4,$5::jsonb,$6,$7,$8,$9,$10,$11,$12)
        ON CONFLICT (id)
        DO UPDATE SET
            strategy_id = EXCLUDED.strategy_id,
            optuna_trial_id = EXCLUDED.optuna_trial_id,
            study_name = EXCLUDED.study_name,
            params_json = EXCLUDED.params_json,
            profit_factor = EXCLUDED.profit_factor,
            total_pips = EXCLUDED.total_pips,
            max_drawdown_pips = EXCLUDED.max_drawdown_pips,
            trade_count = EXCLUDED.trade_count,
            fitness_score = EXCLUDED.fitness_score,
            eliminated_reason = EXCLUDED.eliminated_reason,
            duration_sec = EXCLUDED.duration_sec
        RETURNING id::text
    """

    async def save(self, trial: dict[str, Any]) -> str:
        """Insert or update a trial row.

        Required keys: ``strategy_id``, ``optuna_trial_id``, ``study_name``.
        ``params_json`` defaults to ``{}``; ``id`` defaults to a new UUID.
        """

        require_keys(
            trial,
            {"strategy_id", "optuna_trial_id", "study_name"},
            context="trial_logs",
        )

        trial_id = str(trial.get("id") or uuid.uuid4())
        async def _save(conn):
            saved_id = await conn.fetchval(
                self._UPSERT_SQL,
                as_uuid(trial_id),
                as_uuid(trial["strategy_id"]),
                int(trial["optuna_trial_id"]),
                trial["study_name"],
                as_json(trial.get("params_json", trial.get("params", {}))),
                as_decimal(trial.get("profit_factor")),
                as_decimal(trial.get("total_pips")),
                as_decimal(trial.get("max_drawdown_pips")),
                as_int(trial.get("trade_count")),
                as_decimal(trial.get("fitness_score")),
                trial.get("eliminated_reason"),
                as_decimal(trial.get("duration_sec")),
            )
            return saved_id

        saved_id = await self._run_with_retry(_save, context="save")

        log.info(
            "Trial log saved",
            trial_id=str(saved_id),
            strategy_id=str(trial["strategy_id"]),
            study_name=trial["study_name"],
        )
        return str(saved_id)

    async def create_direct_backtest(
        self,
        *,
        strategy_id: str,
        study_name: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Convenience helper for a non-Optuna full backtest trial."""

        return await self.save(
            {
                "strategy_id": strategy_id,
                "optuna_trial_id": -1,
                "study_name": study_name,
                "params_json": params or {},
            }
        )

    async def update_results(
        self,
        trial_id: str,
        *,
        profit_factor: float | None = None,
        total_pips: float | None = None,
        max_drawdown_pips: float | None = None,
        trade_count: int | None = None,
        fitness_score: float | None = None,
        eliminated_reason: str | None = None,
        duration_sec: float | None = None,
    ) -> None:
        """Update final metrics for an existing trial."""

        async def _update(conn):
            result = await conn.execute(
                """
                UPDATE trial_logs
                SET profit_factor = $2,
                    total_pips = $3,
                    max_drawdown_pips = $4,
                    trade_count = $5,
                    fitness_score = $6,
                    eliminated_reason = $7,
                    duration_sec = $8
                WHERE id = $1
                """,
                as_uuid(trial_id),
                as_decimal(profit_factor),
                as_decimal(total_pips),
                as_decimal(max_drawdown_pips),
                as_int(trade_count),
                as_decimal(fitness_score),
                eliminated_reason,
                as_decimal(duration_sec),
            )
            return result

        result = await self._run_with_retry(_update, context="update_results")

        log.info("Trial results updated", trial_id=trial_id, result=result)

    async def time_trial(self, trial: dict[str, Any]) -> tuple[str, float]:
        """Save a trial with elapsed duration derived from ``started_at``.

        ``started_at`` may be a ``time.monotonic()`` value.  This helper is
        intentionally tiny but keeps callers from repeating timing boilerplate.
        """

        started_at = trial.pop("started_at", None)
        if started_at is not None and trial.get("duration_sec") is None:
            trial["duration_sec"] = time.monotonic() - float(started_at)
        trial_id = await self.save(trial)
        return trial_id, float(trial.get("duration_sec") or 0.0)

    async def get(self, trial_id: str) -> dict[str, Any] | None:
        async def _get(conn):
            row = await conn.fetchrow(
                """
                SELECT id::text, strategy_id::text, optuna_trial_id, study_name,
                       params_json, profit_factor, total_pips, max_drawdown_pips,
                       trade_count, fitness_score, eliminated_reason,
                       duration_sec, created_at
                FROM trial_logs
                WHERE id = $1
                """,
                as_uuid(trial_id),
            )
            return row
        row = await self._run_with_retry(_get, context="get")
        if not row:
            return None
        result = dict(row)
        result["params_json"] = from_json(result.get("params_json"))
        return result


__all__ = ["TrialRepository"]
