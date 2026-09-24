from fastapi import Request

from app.data.market_data import MarketData


def get_market_data(request: Request) -> MarketData:
    """The market data source created at startup (see app.main.create_app)."""
    return request.app.state.market_data
