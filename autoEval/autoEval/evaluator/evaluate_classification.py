from ..python.evaluation.evaluate_classification import evaluate_classification_model
from ..python.reports.classification_report import visualize_classification_predictions
from ..python.fairness.fairness_classification import get_classification_fairness_report
import pandas as pd


def autoevaluate_classification(
    model,
    train_predictions,
    test_predictions,
    feature_columns,
    target,
    prediction_column,
):
    print(f"\n PERFORMANCE METRICS :")
    classification_metrics = evaluate_classification_model(
        train_predictions, test_predictions, target, prediction_column
    )

    print(pd.DataFrame.from_dict(classification_metrics))

    train_predictions[f"{prediction_column}_Probability"] = model.predict_proba(
        train_predictions[feature_columns]
    ).tolist()

    test_predictions[f"{prediction_column}_Probability"] = model.predict_proba(
        test_predictions[feature_columns]
    ).tolist()

    print(f"\n VISUALIZE REPORTS :")
    visualize_classification_predictions(
        train_predictions, test_predictions, target, prediction_column
    )

    train_predictions.drop(f"{prediction_column}_Probability", inplace=True, axis=1)
    test_predictions.drop(f"{prediction_column}_Probability", inplace=True, axis=1)

    print(f"\n FAIRNESS REPORT :")
    fairness_df = get_classification_fairness_report(
        model,
        train_predictions,
        test_predictions,
        feature_columns,
        target,
        prediction_column,
    )

    print(fairness_df.head(fairness_df.shape[0]))
