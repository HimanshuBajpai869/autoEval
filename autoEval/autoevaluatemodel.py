from autoEval.python.reports.regression_report import visualize_regression_predictions
from autoEval.python.evaluation.evaluate_regression import evaluate_regression_model


def autoevaluate_regression(
    train_predictions, test_predictions, target, prediction_column
):
    print(f"\n PERFORMANCE METRICS :")
    print(
        evaluate_regression_model(
            train_predictions, test_predictions, target, prediction_column
        )
    )

    print(f"\n VISUALIZE REPORTS :")
    visualize_regression_predictions(
        train_predictions, test_predictions, target, prediction_column
    )
