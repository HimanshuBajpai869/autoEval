from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_classification_model(
    train_predictions,
    test_predictions,
    target,
    prediction_column,
):
    y_train_actuals = train_predictions[target]
    y_train_predicted = train_predictions[prediction_column]

    y_test_actuals = test_predictions[target]
    y_test_predicted = test_predictions[prediction_column]

    train_metrics = {}
    train_metrics["accuracy_score"] = accuracy_score(y_train_actuals, y_train_predicted)
    train_metrics["precision_score"] = precision_score(
        y_train_actuals, y_train_predicted
    )
    train_metrics["recall_score"] = recall_score(y_train_actuals, y_train_predicted)
    train_metrics["f1_score"] = f1_score(y_train_actuals, y_train_predicted)

    test_metrics = {}
    test_metrics["accuracy_score"] = accuracy_score(y_test_actuals, y_test_predicted)
    test_metrics["precision_score"] = precision_score(y_test_actuals, y_test_predicted)
    test_metrics["recall_score"] = recall_score(y_test_actuals, y_test_predicted)
    test_metrics["f1_score"] = f1_score(y_test_actuals, y_test_predicted)

    return {"train_metrics": train_metrics, "test_metrics": test_metrics}
