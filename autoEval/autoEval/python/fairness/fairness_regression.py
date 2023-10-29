from ...helpers.data_helper import detect_categorical_columns
from ..evaluation.evaluate_regression import evaluate_regression_model


def get_fairness_report(model, train_output, test_output, feature_columns, target):
    categorical_columns = detect_categorical_columns(train_output)
    bias_report = {}

    for categorical_column in categorical_columns:
        print(f"Checking fairness for {categorical_column}")
        unique_values = train_output[categorical_column].unique()

        column_bias_report = {}

        for value in unique_values:
            test_filtered_df = test_output[test_output[categorical_column] == value]
            train_filtered_df = train_output[train_output[categorical_column] == value]

            train_filtered_df["prediction"] = model.predict(
                train_filtered_df[feature_columns]
            )
            test_filtered_df["prediction"] = model.predict(
                test_filtered_df[feature_columns]
            )

            column_bias_report[f"{value}_bias_report"] = evaluate_regression_model(
                train_filtered_df, test_filtered_df, target, "prediction"
            )

        bias_report[f"{categorical_column}_bias_report"] = column_bias_report

    return bias_report
