"""Create market data tables and derived views.

Revision ID: 0001
Revises:
Create Date: 2026-09-24
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# Year-over-year returns compounded from the daily rows on every read. A year is
# partial when the fund has no return in its first or last trading week.
ANNUAL_RETURNS_VIEW = """
CREATE VIEW fund_annual_returns AS
SELECT ticker,
       year,
       EXP(SUM(LN(1 + total_return))) - 1 AS total_return,
       COUNT(*)::integer AS trading_days,
       MIN(trade_date) AS first_date,
       MAX(trade_date) AS last_date,
       NOT (MIN(trade_date) <= make_date(year, 1, 8)
            AND MAX(trade_date) >= make_date(year, 12, 24)) AS is_partial
FROM (
    SELECT ticker, trade_date, total_return,
           EXTRACT(YEAR FROM trade_date)::integer AS year
    FROM fund_daily_returns
) AS daily
GROUP BY ticker, year
"""

# One row per security: history range and compounded returns. YTD covers the
# calendar year of the fund's latest return.
SECURITY_SUMMARY_VIEW = """
CREATE VIEW security_summary AS
WITH latest AS (
    SELECT ticker, MAX(trade_date) AS last_date
    FROM fund_daily_returns
    GROUP BY ticker
)
SELECT s.ticker,
       s.name,
       s.dividend_yield,
       MIN(d.trade_date) AS first_date,
       l.last_date,
       COUNT(*)::integer AS observations,
       EXP(SUM(LN(1 + d.total_return))) - 1 AS since_inception_return,
       EXP(SUM(LN(1 + d.total_return)) FILTER (
           WHERE EXTRACT(YEAR FROM d.trade_date) = EXTRACT(YEAR FROM l.last_date)
       )) - 1 AS ytd_return
FROM securities AS s
JOIN latest AS l ON l.ticker = s.ticker
JOIN fund_daily_returns AS d ON d.ticker = s.ticker
GROUP BY s.ticker, s.name, s.dividend_yield, l.last_date
"""


def upgrade() -> None:
    op.create_table(
        "securities",
        sa.Column("ticker", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("dividend_yield", sa.Double, nullable=True),
        sa.CheckConstraint(
            "dividend_yield >= 0", name="securities_dividend_yield_nonnegative"
        ),
    )
    op.create_table(
        "fund_daily_returns",
        sa.Column(
            "ticker",
            sa.Text,
            sa.ForeignKey("securities.ticker", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("trade_date", sa.Date, primary_key=True),
        sa.Column("total_return", sa.Double, nullable=False),
        sa.CheckConstraint(
            "total_return > -1", name="fund_daily_returns_above_total_loss"
        ),
    )
    op.create_table(
        "factor_daily_returns",
        sa.Column("factor", sa.Text, primary_key=True),
        sa.Column("trade_date", sa.Date, primary_key=True),
        sa.Column("total_return", sa.Double, nullable=False),
        sa.CheckConstraint(
            "factor IN ('momentum', 'value', 'size')",
            name="factor_daily_returns_known_factor",
        ),
    )
    op.create_table(
        "data_loads",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("source_file", sa.Text, nullable=False),
        sa.Column("sha256", sa.Text, nullable=False),
        sa.Column("fund_rows", sa.Integer, nullable=False),
        sa.Column("factor_rows", sa.Integer, nullable=False),
        sa.Column(
            "loaded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.execute(ANNUAL_RETURNS_VIEW)
    op.execute(SECURITY_SUMMARY_VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW security_summary")
    op.execute("DROP VIEW fund_annual_returns")
    op.drop_table("data_loads")
    op.drop_table("factor_daily_returns")
    op.drop_table("fund_daily_returns")
    op.drop_table("securities")
