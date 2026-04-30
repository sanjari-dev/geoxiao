# src/config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # ClickHouse
    CH_HOST: str = 'localhost'
    CH_PORT: int = 8123
    CH_DATABASE: str = 'market_data'
    CH_USER: str = 'default'
    CH_PASSWORD: str = ''

    # PostgreSQL
    PG_HOST: str = 'localhost'
    PG_PORT: int = 5432
    PG_DATABASE: str = 'geoxiao'
    PG_USER: str = 'geoxiao'
    PG_PASSWORD: str = 'secret'
    PG_DSN: str = 'postgresql+asyncpg://geoxiao:secret@localhost:5432/geoxiao'
    PG_DSN_SYNC: str = 'postgresql://geoxiao:secret@localhost:5432/geoxiao'

    # Optuna
    OPTUNA_STORAGE: str = 'postgresql://geoxiao:secret@localhost:5432/geoxiao'

    # Ray
    RAY_ADDRESS: str | None = None

    # Evolution
    N_GENERATIONS: int = 100
    POP_SIZE: int = 50

    # Analytics
    USE_GPU: bool = False

    # Logging
    LOG_FORMAT: str = 'json'   # 'json' | 'console'
    LOG_LEVEL: str = 'INFO'

    model_config = {'env_file': '.env', 'case_sensitive': True}

settings = Settings()
