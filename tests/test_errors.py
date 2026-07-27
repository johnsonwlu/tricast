"""The watchlist acts on these types automatically — dropping a stock it can
never model — so the permanent/transient split has to be exactly right. A
network blip misclassified as permanent means silently deleting the user's
watchlist entries."""

import numpy as np
import pandas as pd
import pytest

from tricast import config, errors
from tricast.quant import montecarlo


def test_permanent_errors_are_flagged_permanent():
    assert errors.is_permanent(errors.UnknownTicker("ZZZZ"))
    assert errors.is_permanent(errors.InsufficientHistory(362, 504, "SNDK"))


@pytest.mark.parametrize("exc", [
    ConnectionError("yahoo timed out"),
    TimeoutError(),
    ValueError("some other problem"),   # plain ValueError must NOT be permanent
    RuntimeError("boom"),
    KeyError("close"),
])
def test_transient_and_unrelated_errors_are_not_permanent(exc):
    """Anything that might succeed on retry must be left alone — acting on it
    would delete a watchlist entry over a temporary outage."""
    assert not errors.is_permanent(exc)


def test_permanent_errors_stay_valueerrors_for_existing_callers():
    """These paths raised ValueError before typed errors existed."""
    assert isinstance(errors.UnknownTicker("X"), ValueError)
    assert isinstance(errors.InsufficientHistory(1, 2), ValueError)


def test_insufficient_history_friendly_message_is_beginner_readable():
    msg = errors.InsufficientHistory(have=362, need=504, ticker="SNDK").friendly()
    assert "SNDK" in msg
    assert "362" in msg
    assert "newly" in msg or "track record" in msg
    for jargon in ("lookback", "MIN_HISTORY_DAYS", "504 days"):
        assert jargon not in msg


def test_insufficient_history_without_ticker_still_reads_sensibly():
    msg = errors.InsufficientHistory(have=100, need=504).friendly()
    assert msg.startswith("This stock")


def test_unknown_ticker_friendly_message_suggests_causes():
    msg = errors.UnknownTicker("ZZZZ").friendly()
    assert "ZZZZ" in msg
    assert "mistyped" in msg or "delisted" in msg


def test_simulate_raises_typed_insufficient_history():
    """The real trigger: a stock with under two years of history."""
    short = pd.Series(100 * np.exp(np.cumsum(
        np.random.default_rng(0).normal(0, 0.01, 362))))
    with pytest.raises(errors.InsufficientHistory) as excinfo:
        montecarlo.simulate(short)
    assert excinfo.value.have == 362
    assert excinfo.value.need == config.MIN_HISTORY_DAYS
    assert errors.is_permanent(excinfo.value)
