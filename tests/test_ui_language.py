"""The plain-language layer is user-facing copy, but it's still logic: the
wrong branch here tells a beginner the opposite of the truth. These tests pin
the mappings that a careless edit could silently invert."""

import pytest

from tricast import ui


def test_every_scenario_has_plain_label_term_and_explanation():
    for name in ("bull", "base", "bear"):
        label, term, why = ui.SCENARIOS[name]
        assert label and term and why
        # the everyday label must not just repeat the jargon
        assert term not in label.lower()
        assert ui.scenario_label(name) == label
        assert ui.scenario_term(name) == term
        assert ui.scenario_help(name) == why


def test_scenario_labels_convey_direction_without_finance_words():
    assert "well" in ui.scenario_label("bull")
    assert "poorly" in ui.scenario_label("bear")
    assert "likely" in ui.scenario_label("base")
    joined = " ".join(ui.scenario_label(n) for n in ("bull", "base", "bear")).lower()
    for jargon in ("bull", "bear", "percentile", "p25", "band"):
        assert jargon not in joined


@pytest.mark.parametrize("pct,expected", [
    (0.0, "roughly flat"), (0.1, "roughly flat"), (-0.2, "roughly flat"),
    (0.5, "up a little"), (-0.5, "down a little"),
    (2.0, "up moderately"), (-2.0, "down moderately"),
    (5.0, "up a lot"), (-7.5, "down a lot"),
])
def test_change_words(pct, expected):
    assert ui.change_words(pct) == expected


def test_change_words_direction_is_never_inverted():
    for pct in (0.4, 1.5, 3.0, 25.0):
        assert ui.change_words(pct).startswith("up")
        assert ui.change_words(-pct).startswith("down")


def test_change_words_scales_with_the_time_window():
    """A 4% day is dramatic; a 4% month is ordinary. Describing both the same
    way misleads a beginner, which is exactly what the first version did."""
    assert ui.change_words(3.9, "day") == "up a lot"
    assert ui.change_words(3.9, "month") == "up a little"
    assert ui.change_words(2.0, "year") == "roughly flat"
    assert ui.change_words(2.0, "day") == "up moderately"
    # and the extremes still read as extreme at every scale
    assert ui.change_words(45, "day") == ui.change_words(45, "year") == "up a lot"


def test_change_words_unknown_scale_falls_back_to_daily():
    assert ui.change_words(2.0, "fortnight") == ui.change_words(2.0, "day")


@pytest.mark.parametrize("pct,expected", [
    (90, "very likely"), (65, "very likely"),
    (50, "about as likely as not"), (45, "about as likely as not"),
    (30, "possible"), (25, "possible"),
    (10, "unlikely"), (0, "unlikely"),
])
def test_chance_words(pct, expected):
    assert ui.chance_words(pct) == expected


def test_chance_words_is_monotone():
    """Wording must never get more confident as the probability falls."""
    order = ["unlikely", "possible", "about as likely as not", "very likely"]
    ranks = [order.index(ui.chance_words(p)) for p in range(0, 101, 5)]
    assert ranks == sorted(ranks)


def test_plain_advice_covers_every_advice_value():
    for advice in ("buy", "hold", "avoid"):
        sentence = ui.plain_advice(advice)
        assert sentence.endswith(".")
        assert advice.upper() not in sentence      # explains, doesn't just shout
    assert ui.plain_advice("nonsense") == "No recommendation yet."


def test_advice_pill_tone_matches_meaning():
    assert ui.ADVICE_TONES["buy"] == "good"
    assert ui.ADVICE_TONES["avoid"] == "poor"
    assert "Worth buying" in ui.advice_pill("buy")
    assert "avoid" in ui.advice_pill("avoid").lower()


def test_risk_label_stays_chip_sized():
    """It renders inside a narrow card. The first version put a whole sentence
    in a nowrap chip, which ran outside the panel."""
    for label in ("strong", "fair", "weak", "poor"):
        text, tone = ui.risk_label({"label": label, "prob_loss_pct": 30})
        assert len(text) <= 22, f"{text!r} is too long for the chip"
        assert len(text.split()) <= 4
        assert tone in ui.TONES


def test_long_pill_opts_into_wrapping():
    assert "tc-pill-wrap" in ui.pill("a very long status sentence", wrap=True)
    assert "tc-pill-wrap" not in ui.pill("Short")


def test_loss_note_reads_as_a_sentence():
    assert ui.loss_note({"prob_loss_pct": 28}) == "Loses money in 28% of simulations"


def test_risk_sentence_tone_tracks_the_label():
    _, good = ui.risk_sentence({"label": "strong", "prob_loss_pct": 12})
    _, bad = ui.risk_sentence({"label": "poor", "prob_loss_pct": 55})
    assert good == "good" and bad == "poor"
    sentence, _ = ui.risk_sentence({"label": "weak", "prob_loss_pct": 40})
    assert "40%" in sentence


def test_risk_sentence_survives_missing_fields():
    """Risk scoring lands in a separate change; the UI must not explode if the
    block is partial."""
    sentence, tone = ui.risk_sentence({})
    assert sentence and tone in ui.TONES


def test_regime_plain_direction():
    good, tone_g = ui.regime_plain({"regime": "Expansionary", "score": 0.5})
    bad, tone_b = ui.regime_plain({"regime": "Contractionary", "score": -0.5})
    neutral, _ = ui.regime_plain({"regime": "Neutral", "score": 0.0})
    assert tone_g == "good" and "supportive" in good
    assert tone_b == "poor" and "headwind" in bad
    assert "neither" in neutral


def test_pill_renders_known_and_unknown_tones():
    assert "background:#E9F4EF" in ui.pill("ok", "good").replace(" ", "")
    # unknown tone must fall back rather than raise
    assert ui.pill("x", "not-a-tone")


def test_help_text_is_present_and_jargon_free_where_it_counts():
    for key in ("sharpe", "prob_loss", "cvar", "beta", "chance"):
        assert key in ui.HELP
        assert len(ui.HELP[key]) > 40          # a real explanation, not a label
    # the reward-for-risk tooltip must avoid naming the ratio it explains
    assert "sharpe" not in ui.HELP["sharpe"].lower()
