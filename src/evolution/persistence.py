# src/evolution/persistence.py
# Referensi: Blueprint §6.3

from __future__ import annotations
import json
import pickle
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
import structlog

log = structlog.get_logger(__name__)


@dataclass
class EvolutionCheckpoint:
    """
    Snapshot lengkap state evolusi pada satu titik waktu.
    Berisi semua informasi yang dibutuhkan untuk resume.
    """
    run_id: str
    generation: int
    population: list[dict]      # list of StrategyDNA.to_db_dict()
    hall_of_fame: list[dict]    # Top 10 individu terbaik sepanjang masa
    optuna_storage: str         # URI storage Optuna (PG DSN)
    config_snapshot: dict       # Settings.model_dump() saat checkpoint
    total_evaluated: int = 0
    total_passed: int = 0
    saved_at: str = field(default=None)

    def __post_init__(self) -> None:
        if self.saved_at is None:
            self.saved_at = datetime.now(timezone.utc).isoformat()


class CheckpointManager:
    """
    Simpan dan load EvolutionCheckpoint.

    File naming convention:
    - JSON: {run_id}_gen{N:04d}.json  (per-generasi, untuk audit trail)
    - PKL:  {run_id}_latest.pkl       (latest, untuk resume)
    """

    CHECKPOINT_DIR = Path('./checkpoints')

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: EvolutionCheckpoint) -> None:
        """
        Simpan checkpoint ke JSON + pickle.
        Operasi dilakukan secara atomic: tulis ke temp file, lalu rename.
        """
        gen = checkpoint.generation

        # ── JSON audit trail ──────────────────────────────────────────
        json_path = self.CHECKPOINT_DIR / f'{self.run_id}_gen{gen:04d}.json'
        json_tmp = json_path.with_suffix('.tmp')
        with open(json_tmp, 'w', encoding='utf-8') as f:
            json.dump(asdict(checkpoint), f, indent=2, default=str)
        json_tmp.rename(json_path)

        # ── Pickle untuk resume ───────────────────────────────────────
        pkl_path = self.CHECKPOINT_DIR / f'{self.run_id}_latest.pkl'
        pkl_tmp = pkl_path.with_suffix('.tmp')
        with open(pkl_tmp, 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
        pkl_tmp.rename(pkl_path)

        log.info('Checkpoint saved',
                 run_id=self.run_id,
                 generation=gen,
                 population_size=len(checkpoint.population),
                 hof_size=len(checkpoint.hall_of_fame),
                 path=str(json_path))

    def load_latest(self) -> EvolutionCheckpoint | None:
        """
        Load checkpoint terbaru untuk run_id ini.

        Returns:
            EvolutionCheckpoint jika ada, None jika tidak ada checkpoint.
        """
        pkl_path = self.CHECKPOINT_DIR / f'{self.run_id}_latest.pkl'

        if not pkl_path.exists():
            log.warning('No checkpoint found', run_id=self.run_id, path=str(pkl_path))
            return None

        try:
            with open(pkl_path, 'rb') as f:
                checkpoint: EvolutionCheckpoint = pickle.load(f)
            log.info('Checkpoint loaded',
                     run_id=self.run_id,
                     generation=checkpoint.generation,
                     saved_at=checkpoint.saved_at)
            return checkpoint
        except Exception as e:
            log.error('Failed to load checkpoint', run_id=self.run_id, error=str(e))
            return None

    def list_checkpoints(self) -> list[Path]:
        """List semua JSON checkpoint untuk run_id ini, sorted ascending."""
        return sorted(self.CHECKPOINT_DIR.glob(f'{self.run_id}_gen*.json'))
