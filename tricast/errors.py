"""Typed failures, split by whether retrying could ever help.

The watchlist needs this distinction to decide what to do with a stock it can't
load. A newly-listed company that doesn't have two years of trading history yet
will *never* load, no matter how many times you refresh — leaving it sitting in
the list with an error is just clutter. A network blip looks identical from the
outside but fixes itself, and quietly dropping a stock because Yahoo timed out
would be data loss.

So: permanent failures are safe to act on automatically, transient ones are not.
Both carry a `friendly()` message written for someone who won't know what a
"lookback window" is.
"""


class TricastError(Exception):
    """Base for errors this app raises deliberately."""

    def friendly(self) -> str:
        return str(self)


class PermanentTickerError(TricastError, ValueError):
    """A stock this app structurally cannot model. Retrying will not help.

    Subclasses ValueError because that's what these paths raised before typed
    errors existed, so existing callers and tests keep working.
    """


class UnknownTicker(PermanentTickerError):
    def __init__(self, ticker: str):
        self.ticker = ticker
        super().__init__(f"No price data available for {ticker!r}")

    def friendly(self) -> str:
        return (f"{self.ticker} doesn't match any stock we can get prices for. "
                "It may have been delisted, renamed, or mistyped.")


class InsufficientHistory(PermanentTickerError):
    def __init__(self, have: int, need: int, ticker: str | None = None):
        self.have, self.need, self.ticker = have, need, ticker
        who = f"{ticker}: " if ticker else ""
        super().__init__(f"{who}Need at least {need} days of history, got {have}")

    def friendly(self) -> str:
        who = self.ticker or "This stock"
        years = self.need / 252
        return (f"{who} has only {self.have} days of trading history — about "
                f"{self.have / 252:.1f} years. The forecast needs at least "
                f"{years:.0f} years to be worth anything, so it's too newly "
                "listed to model. Try again once it has a longer track record.")


def is_permanent(exc: BaseException) -> bool:
    """True if retrying could never succeed, so it's safe to act on the failure
    (drop the ticker) rather than leave it sitting there broken."""
    return isinstance(exc, PermanentTickerError)
