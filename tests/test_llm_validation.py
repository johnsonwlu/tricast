from stock_scenarios.llm.analyst import enforce_bounds, inputs_hash


def test_within_bounds_passes_through():
    tilted = {"bear": 29, "base": 50, "bull": 21}
    proposed = {"bear": 25, "base": 52, "bull": 23}
    assert enforce_bounds(proposed, tilted) == proposed


def test_out_of_bounds_clamped_residual_to_base():
    # plan's regression case: mocked 60/30/10 vs prior 29/50/21
    tilted = {"bear": 29, "base": 50, "bull": 21}
    proposed = {"bear": 60, "base": 30, "bull": 10}
    result = enforce_bounds(proposed, tilted)
    assert sum(result.values()) == 100
    for k in result:
        assert abs(result[k] - tilted[k]) <= 10


def test_unrecoverable_falls_back_to_priors():
    tilted = {"bear": 25, "base": 50, "bull": 25}
    # both tails pinned high forces base far below its bound
    proposed = {"bear": 90, "base": 0, "bull": 90}
    assert enforce_bounds(proposed, tilted) == tilted


def test_exact_priors_unchanged():
    tilted = {"bear": 15, "base": 50, "bull": 35}
    assert enforce_bounds(dict(tilted), tilted) == tilted


def test_inputs_hash_deterministic_and_order_insensitive():
    a = inputs_hash({"x": 1, "y": [1, 2]})
    b = inputs_hash({"y": [1, 2], "x": 1})
    assert a == b
    assert a != inputs_hash({"x": 2, "y": [1, 2]})
