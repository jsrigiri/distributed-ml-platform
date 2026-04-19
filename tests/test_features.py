from pipelines.online_features import OnlineFeatureBuilder


def test_online_features_build_after_warmup():
    builder = OnlineFeatureBuilder(warmup=5)

    events = [
        {"user_id": 1, "event_value": 1.0, "amount": 10.0, "hour": 9, "is_mobile": 1},
        {"user_id": 1, "event_value": 1.2, "amount": 11.0, "hour": 10, "is_mobile": 0},
        {"user_id": 1, "event_value": 0.9, "amount": 12.0, "hour": 11, "is_mobile": 1},
        {"user_id": 1, "event_value": 1.1, "amount": 9.5, "hour": 12, "is_mobile": 1},
    ]

    for e in events:
        out = builder.update(**e)
        assert out is None

    out = builder.update(
        user_id=1,
        event_value=1.3,
        amount=10.5,
        hour=13,
        is_mobile=0,
    )

    assert out is not None
    assert "lag_event_value_1" in out
    assert "rolling_mean_event_value_5" in out
    assert "rolling_std_event_value_5" in out
    assert "events_seen" in out


def test_online_features_are_isolated_by_user():
    builder = OnlineFeatureBuilder(warmup=3)

    assert builder.update(1, 1.0, 10.0, 9, 1) is None
    assert builder.update(2, 2.0, 20.0, 10, 0) is None
    assert builder.update(1, 1.1, 11.0, 11, 1) is None
    assert builder.update(2, 2.1, 21.0, 12, 1) is None

    out1 = builder.update(1, 1.2, 12.0, 13, 0)
    out2 = builder.update(2, 2.2, 22.0, 14, 1)

    assert out1 is not None
    assert out2 is not None
    assert out1["event_value"] != out2["event_value"]