"""Tables and views used by queries.

The schema itself is created by migrations/versions/0001_create_market_data.py.
"""

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    func,
)

metadata = MetaData()

securities = Table(
    "securities",
    metadata,
    Column("ticker", Text, primary_key=True),
    Column("name", Text, nullable=False),
    Column("dividend_yield", Double, nullable=True),  # NULL = blank in the source
    CheckConstraint(
        "dividend_yield >= 0", name="securities_dividend_yield_nonnegative"
    ),
)

fund_daily_returns = Table(
    "fund_daily_returns",
    metadata,
    Column(
        "ticker",
        Text,
        ForeignKey("securities.ticker", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("trade_date", Date, primary_key=True),
    Column("total_return", Double, nullable=False),
    # A traded fund cannot lose 100% in a day and keep trading.
    CheckConstraint("total_return > -1", name="fund_daily_returns_above_total_loss"),
)

factor_daily_returns = Table(
    "factor_daily_returns",
    metadata,
    Column("factor", Text, primary_key=True),
    Column("trade_date", Date, primary_key=True),
    Column("total_return", Double, nullable=False),
    CheckConstraint(
        "factor IN ('momentum', 'value', 'size')",
        name="factor_daily_returns_known_factor",
    ),
)

data_loads = Table(
    "data_loads",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("source_file", Text, nullable=False),
    Column("sha256", Text, nullable=False),
    Column("fund_rows", Integer, nullable=False),
    Column("factor_rows", Integer, nullable=False),
    Column(
        "loaded_at", DateTime(timezone=True), nullable=False, server_default=func.now()
    ),
)

# Views live in their own MetaData so they're never mistaken for tables.
views = MetaData()

fund_annual_returns = Table(
    "fund_annual_returns",
    views,
    Column("ticker", Text),
    Column("year", Integer),
    Column("total_return", Double),
    Column("trading_days", Integer),
    Column("first_date", Date),
    Column("last_date", Date),
    Column("is_partial", Boolean),
)

security_summary = Table(
    "security_summary",
    views,
    Column("ticker", Text),
    Column("name", Text),
    Column("dividend_yield", Double),
    Column("first_date", Date),
    Column("last_date", Date),
    Column("observations", Integer),
    Column("since_inception_return", Double),
    Column("ytd_return", Double),
)
