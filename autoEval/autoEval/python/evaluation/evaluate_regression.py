from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)


def evaluate_regression_model(
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
    train_metrics["r2_score"] = r2_score(y_train_actuals, y_train_predicted)
    train_metrics["mae_score"] = mean_absolute_error(y_train_actuals, y_train_predicted)
    train_metrics["mse_score"] = mean_squared_error(y_train_actuals, y_train_predicted)
    train_metrics["mape_score"] = mean_absolute_percentage_error(
        y_train_actuals, y_train_predicted
    )

    test_metrics = {}
    test_metrics["r2_score"] = r2_score(y_test_actuals, y_test_predicted)
    test_metrics["mae_score"] = mean_absolute_error(y_test_actuals, y_test_predicted)
    test_metrics["mse_score"] = mean_squared_error(y_test_actuals, y_test_predicted)
    test_metrics["mape_score"] = mean_absolute_percentage_error(
        y_test_actuals, y_test_predicted
    )

    return {"train_metrics": train_metrics, "test_metrics": test_metrics}
