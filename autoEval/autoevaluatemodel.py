from .autoEval.evaluator.evaluate_regression import autoevaluate_regression
from .autoEval.helpers.enumerators import ModellingTaskType
import warnings

warnings.filterwarnings("ignore")


def auto_evaluate_model(
    model,
    train_predictions,
    test_predictions,
    feature_columns,
    target,
    prediction_column,
    modelling_task_type="regression",
):
    """
    The function helps evaluate the model and provides below reports :

    1. Model Performance Metrics
    2. Model Performance Report
    3. Model Fairness Report

    Inputs :

    model : object
        The model instance from sklearn/spark or pipeline object.

    train_predictions : dataframe
        The dataframe with feature columns, target, predicted column with train data.

    test_predictions : dataframe
        The dataframe with feature columns, target, predicted column with test data.

    feature_columns : list
        The list of feature columns used for training the model.

    target : str
        The name of the actual target column available in the dataframe.

    prediction_column : str
        The name of the predicted column containing the model predictions.

    modelling_task_type : str, optional
        The name of modelling task type. The default value is "regression". It should be one of the below -

        1. Regression
        2. Classification
        3. Forecasting

    Returns : None
        The function performs the auto evaluation of the model and dislays the result.
    """

    if modelling_task_type.lower() == ModellingTaskType.Regression.value.lower():
        autoevaluate_regression(
            model,
            train_predictions,
            test_predictions,
            feature_columns,
            target,
            prediction_column,
        )
    else:
        raise Exception(
            f"The input modelling task type {modelling_task_type} is currently not supported."
        )
