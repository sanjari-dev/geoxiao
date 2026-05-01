# src/data/migrations/env.py
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from src.data.repositories.base import postgres_sync_dsn

config = context.config
# Gunakan DSN sinkron (non-asyncpg) untuk keperluan migrasi Alembic
config.set_main_option('sqlalchemy.url', postgres_sync_dsn())

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    pass # Offline mode disingkirkan untuk simplifikasi
else:
    run_migrations_online()
