from ...helpers.data_helper import detect_categorical_columns, flatten
from ..evaluation.evaluate_classification import evaluate_classification_model


def get_classification_fairness_report(
    model, train_output, test_output, feature_columns, target, prediction_column
):
    categorical_columns = detect_categorical_columns(train_output)
    bias_report = {}

    for categorical_column in categorical_columns:
        if categorical_column == prediction_column or categorical_column == target:
            continue
        print(f"Checking fairness for {categorical_column}")
        unique_values = train_output[categorical_column].unique()

        column_bias_report = {}

        for value in unique_values:
            test_filtered_df = test_output[test_output[categorical_column] == value]
            train_filtered_df = train_output[train_output[categorical_column] == value]

            train_filtered_df[prediction_column] = model.predict(
                train_filtered_df[feature_columns]
            )
            test_filtered_df[prediction_column] = model.predict(
                test_filtered_df[feature_columns]
            )

            column_bias_report[f"{value}"] = evaluate_classification_model(
                train_filtered_df, test_filtered_df, target, prediction_column
            )

        bias_report[f"{categorical_column}_bias_report"] = column_bias_report

    return flatten(bias_report)
