"""Connection settings for the PostgreSQL market data store."""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

# Matches docker-compose.yml. Port 5433 avoids clashing with a local PostgreSQL.
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://portfolio:portfolio@localhost:5433/portfolio"
)


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def create_database_engine(url: str | None = None) -> Engine:
    """Pooled engine; pre-ping replaces connections dropped by a database restart."""
    return create_engine(url or database_url(), pool_pre_ping=True)
