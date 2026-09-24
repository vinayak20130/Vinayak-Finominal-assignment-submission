"""Alembic environment: runs migrations against DATABASE_URL."""

from logging.config import fileConfig

from alembic import context

from app.data.database import create_database_engine
from app.data.tables import metadata

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def run_migrations_online() -> None:
    engine = create_database_engine()
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=metadata)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    raise SystemExit("Offline migrations are not supported; set DATABASE_URL.")
run_migrations_online()
