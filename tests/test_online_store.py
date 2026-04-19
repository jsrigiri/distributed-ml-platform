from pipelines.online_features import OnlineFeatureBuilder


def test_online_features_build_after_warmup():
    builder = OnlineFeatureBuilder(warmup=3)

    assert builder.update(1, 1.0, 10.0, 9, 1) is None
    assert builder.update(1, 1.1, 11.0, 10, 0) is None

    out = builder.update(1, 1.2, 12.0, 11, 1)
    assert out is not None
    assert "event_value" in out
    assert "lag_event_value_1" in out
    assert "rolling_mean_event_value_5" in out