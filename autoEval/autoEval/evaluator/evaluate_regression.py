from ..python.reports.regression_report import visualize_regression_predictions
from ..python.evaluation.evaluate_regression import evaluate_regression_model
from ..python.fairness.fairness_regression import get_regression_fairness_report
import pandas as pd


def autoevaluate_regression(
    model,
    train_predictions,
    test_predictions,
    feature_columns,
    target,
    prediction_column,
):
    print(f"\n PERFORMANCE METRICS :")
    regression_metrics = evaluate_regression_model(
        train_predictions, test_predictions, target, prediction_column
    )

    print(pd.DataFrame.from_dict(regression_metrics))

    print(f"\n VISUALIZE REPORTS :")
    visualize_regression_predictions(
        train_predictions, test_predictions, target, prediction_column
    )

    print(f"\n FAIRNESS REPORT :")
    fairness_df = get_regression_fairness_report(
        model,
        train_predictions,
        test_predictions,
        feature_columns,
        target,
        prediction_column,
    )

    print(fairness_df)  # .head(fairness_df.shape[0]))
