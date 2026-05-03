def should_promote_classifier(
    champion_metrics: dict,
    challenger_metrics: dict,
) -> tuple[bool, str]:
    champion_acc = float(champion_metrics.get("accuracy", 0.0))
    challenger_acc = float(challenger_metrics.get("accuracy", 0.0))

    champion_f1 = float(champion_metrics.get("f1", 0.0))
    challenger_f1 = float(challenger_metrics.get("f1", 0.0))

    min_accuracy_lift = 0.01
    max_f1_drop = 0.005

    if challenger_acc < champion_acc + min_accuracy_lift:
        return (
            False,
            f"challenger accuracy {challenger_acc:.4f} did not beat champion "
            f"accuracy {champion_acc:.4f} by at least {min_accuracy_lift:.4f}",
        )

    if challenger_f1 < champion_f1 - max_f1_drop:
        return (
            False,
            f"challenger f1 {challenger_f1:.4f} is too far below champion "
            f"f1 {champion_f1:.4f}",
        )

    return True, "challenger outperformed champion"


def should_promote_regressor(
    champion_metrics: dict,
    challenger_metrics: dict,
) -> tuple[bool, str]:
    champion_rmse = float(champion_metrics.get("rmse", float("inf")))
    challenger_rmse = float(challenger_metrics.get("rmse", float("inf")))

    min_rmse_improvement = 0.01

    if challenger_rmse > champion_rmse - min_rmse_improvement:
        return (
            False,
            f"challenger rmse {challenger_rmse:.4f} did not improve enough over "
            f"champion rmse {champion_rmse:.4f}",
        )

    return True, "challenger outperformed champion"