from autoEval.python.reports.regression_report import visualize_regression_predictions
from autoEval.python.evaluation.evaluate_regression import evaluate_regression_model
from autoEval.python.fairness.fairness_regression import get_fairness_report
import warnings

warnings.filterwarnings("ignore")


def autoevaluate_regression(
    model,
    train_predictions,
    test_predictions,
    feature_columns,
    target,
    prediction_column,
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

    print(f"\n FAIRNESS REPORT :")
    print(
        get_fairness_report(
            model, train_predictions, test_predictions, feature_columns, target
        )
    )
