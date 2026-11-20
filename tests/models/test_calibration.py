from market_lab.models.calibration import ProbabilityCalibrator, ResidualIntervalCalibrator


def test_probability_calibration_is_bounded_and_uses_training_rows_only() -> None:
    calibrator = ProbabilityCalibrator().fit(
        raw_scores=[-2.0, -1.0, 0.5, 2.0],
        labels=[0, 0, 1, 1],
    )

    probabilities = calibrator.predict([-10.0, 0.0, 10.0])

    assert calibrator.fitted_count == 4
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert probabilities == sorted(probabilities)


def test_residual_interval_calibration_has_expected_training_coverage() -> None:
    calibrator = ResidualIntervalCalibrator(alpha=0.2).fit(
        predictions=[0.0, 0.0, 0.0, 0.0, 0.0],
        actuals=[-0.04, -0.01, 0.0, 0.02, 0.05],
    )

    lower, upper = calibrator.interval([0.01])

    assert calibrator.fitted_count == 5
    assert lower[0] <= 0.01 <= upper[0]
    assert upper[0] - lower[0] > 0
