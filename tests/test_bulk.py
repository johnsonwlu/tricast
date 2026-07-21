from tricast import bulk


def _report(ticker, advice="buy", cost=0.0):
    return {"ticker": ticker, "analysis": {"advice": advice, "cost_usd": cost}}


def test_analyze_many_isolates_failures(monkeypatch):
    def fake_build(ticker, run_llm=True, db_path=None):
        if ticker == "BAD":
            raise ValueError("no data")
        return _report(ticker, advice="hold" if ticker == "MSFT" else "buy")

    monkeypatch.setattr(bulk.pipeline, "build_report", fake_build)
    seen = []
    results = bulk.analyze_many(["AAPL", "BAD", "MSFT"],
                                progress_cb=lambda *a: seen.append(a[3]["ticker"]))

    assert [r["ticker"] for r in results] == ["AAPL", "BAD", "MSFT"]
    assert [r["ok"] for r in results] == [True, False, True]
    assert results[1]["error"] == "no data"
    assert seen == ["AAPL", "BAD", "MSFT"]  # progress fired for every ticker


def test_summarize_counts_and_cost(monkeypatch):
    def fake_build(ticker, run_llm=True, db_path=None):
        return {"AAPL": _report("AAPL", "buy", 0.02),
                "MSFT": _report("MSFT", "hold", 0.03),
                "KO": _report("KO", "buy", 0.0)}[ticker]

    monkeypatch.setattr(bulk.pipeline, "build_report", fake_build)
    s = bulk.summarize(bulk.analyze_many(["AAPL", "MSFT", "KO"]))
    assert s["ok"] == 3 and s["failed"] == 0
    assert s["advice_counts"] == {"buy": 2, "hold": 1}
    assert s["total_cost_usd"] == 0.05


def test_no_llm_row_has_no_advice(monkeypatch):
    monkeypatch.setattr(bulk.pipeline, "build_report",
                        lambda t, run_llm=True, db_path=None: {"ticker": t, "analysis": None})
    results = bulk.analyze_many(["AAPL"], run_llm=False)
    assert results[0]["ok"] and results[0]["advice"] is None


def test_analyze_watchlist_uses_store(monkeypatch):
    monkeypatch.setattr(bulk.store, "watchlist_all", lambda db_path=None: ["AAPL", "MSFT"])
    monkeypatch.setattr(bulk.pipeline, "build_report",
                        lambda t, run_llm=True, db_path=None: _report(t))
    results = bulk.analyze_watchlist()
    assert [r["ticker"] for r in results] == ["AAPL", "MSFT"]
