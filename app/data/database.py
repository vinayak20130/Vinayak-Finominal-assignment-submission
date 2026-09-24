"""Connection settings for the PostgreSQL market data store."""

import os

from sqlalchemy import create_engine, make_url
from sqlalchemy.engine import Engine

CONNECT_TIMEOUT_SECONDS = 5
POOL_TIMEOUT_SECONDS = 5

# Matches docker-compose.yml. Port 5433 avoids clashing with a local PostgreSQL.
DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://portfolio:portfolio@localhost:5433/portfolio"
)


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def create_database_engine(url: str | None = None) -> Engine:
    """Pooled engine whose every wait is bounded, so a hung database fails fast
    (as a 503) instead of blocking requests. Pre-ping replaces connections dropped
    by a database restart. Timeouts already present in the URL are kept."""
    target = make_url(url or database_url())
    defaults = {
        "connect_timeout": str(CONNECT_TIMEOUT_SECONDS),
        # Also bounds a pooled connection whose server stopped answering (Linux).
        "tcp_user_timeout": str(CONNECT_TIMEOUT_SECONDS * 1000),
    }
    target = target.update_query_dict({**defaults, **target.query})
    return create_engine(target, pool_pre_ping=True, pool_timeout=POOL_TIMEOUT_SECONDS)
