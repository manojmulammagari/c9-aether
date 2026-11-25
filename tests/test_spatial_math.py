import math
from ml_pipeline.spatial_math import (
    FEATURE_ORDER,
    PlayerState,
    SmokeOrb,
    compute_operator_credit_risk,
    extract_round_features,
    feature_dict_to_vector,
    has_line_of_sight,
)


def test_wall_blocks_operator_line_of_sight() -> None:
    assert has_line_of_sight(22.0, 22.0, 22.0, 50.0) is False
    assert has_line_of_sight(12.0, 22.0, 30.0, 22.0) is True


def test_counterfactual_smoke_breaks_operator_sightline() -> None:
    players = [
        PlayerState("atk1", "ATK", x=20.0, y=18.0, alive=True),
        PlayerState("def_op", "DEF", x=30.0, y=18.0, alive=True, has_operator=True, loadout_value=5700.0),
    ]
    no_smoke = extract_round_features(players, smokes=[])
    with_smoke = extract_round_features(players, smokes=[SmokeOrb(x=25.0, y=18.0, radius=4.0)])

    assert no_smoke["op_targets_in_sight"] == 1.0
    assert with_smoke["op_targets_in_sight"] == 0.0
    assert with_smoke["op_exposed_area_ratio"] < no_smoke["op_exposed_area_ratio"]


def test_nan_telemetry_safety_and_vector_contract() -> None:
    players = [
        PlayerState("atk_ok", "ATK", x=25.0, y=25.0, alive=True, loadout_value=float("nan")),
        PlayerState("atk_bad", "ATK", x=float("nan"), y=30.0, alive=True),
        PlayerState("def_ok", "DEF", x=78.0, y=25.0, alive=True, loadout_value=4500.0),
    ]
    feats = extract_round_features(players, time_remaining_sec=float("nan"))
    vec = feature_dict_to_vector(feats)

    assert vec.shape == (len(FEATURE_ORDER),)
    assert feats["atk_alive"] == 1.0
    assert not any(math.isnan(v) for v in vec)
    assert compute_operator_credit_risk(0.60, op_alive_def=True) == 2820.0
