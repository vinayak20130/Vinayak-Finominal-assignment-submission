import pytest
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from app.data.database import create_database_engine
from app.data.postgres import PostgresMarketData
from app.domain.errors import DataUnavailableError


class ExhaustedPoolEngine:
    """Engine stand-in whose connection pool is exhausted."""

    def connect(self):
        raise PoolTimeoutError("QueuePool limit reached")


def test_pool_timeout_is_data_unavailable():
    with pytest.raises(DataUnavailableError):
        PostgresMarketData(ExhaustedPoolEngine()).securities()


def test_engine_bounds_every_wait():
    engine = create_database_engine("postgresql+psycopg://u:p@localhost:1/db")
    try:
        assert engine.pool.timeout() == 5
        assert engine.url.query["connect_timeout"] == "5"
        assert engine.url.query["tcp_user_timeout"] == "5000"
    finally:
        engine.dispose()


def test_timeouts_in_database_url_are_kept():
    url = "postgresql+psycopg://u:p@localhost:1/db?connect_timeout=2"
    engine = create_database_engine(url)
    try:
        assert engine.url.query["connect_timeout"] == "2"
    finally:
        engine.dispose()
