"""Alembic environment supporting SQLite local mode and optional PostgreSQL DATABASE_URL."""

from __future__ import annotations

from logging.config import fileConfig
from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import settings

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

db_url = settings.database_url.strip() or f"sqlite:///{settings.fortivex_db_path}"
config.set_main_option("sqlalchemy.url", db_url)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
