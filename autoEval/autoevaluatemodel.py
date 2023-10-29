from autoEval.evaluator.evaluate_regression import autoevaluate_regression
from autoEval.helpers.enumerators import ModellingTaskType


def auto_evaluate_model(
    model,
    train_predictions,
    test_predictions,
    feature_columns,
    target,
    prediction_column,
    modelling_task_type="regression",
):
    if modelling_task_type.lower() == ModellingTaskType.Regression.value.lower():
        autoevaluate_regression(
            model,
            train_predictions,
            test_predictions,
            feature_columns,
            target,
            prediction_column,
        )
