# src/data/retry.py
# Referensi: Blueprint §6.2

import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)
import structlog

log = structlog.get_logger(__name__)
_stdlib_log = logging.getLogger(__name__)


# ── Decorator untuk semua operasi DB ─────────────────────────────────────
# Konfigurasi:
# - 5 attempt maksimum
# - Backoff: mulai 2 detik, maksimum 30 detik, multiplier 1
# - Retry pada: ConnectionError, TimeoutError, OSError
# - Log WARNING sebelum setiap retry

db_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    before_sleep=before_sleep_log(_stdlib_log, logging.WARNING),
    reraise=True,   # Raise exception asli setelah semua attempt habis
)


# ── Decorator untuk operasi ClickHouse (lebih aggressive) ────────────────
clickhouse_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=1, max=20),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
    before_sleep=before_sleep_log(_stdlib_log, logging.WARNING),
    reraise=True,
)
