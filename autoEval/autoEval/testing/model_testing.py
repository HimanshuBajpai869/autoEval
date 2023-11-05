from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite
from ..helpers.data_helper import detect_categorical_columns


def perform_model_testing(
    model, train_predictions, test_predictions, feature_columns, target_column
):
    df_train = train_predictions[feature_columns + [target_column]]
    df_test = test_predictions[feature_columns + [target_column]]
    cat_columns = detect_categorical_columns(df_train[feature_columns])

    # Prepare data for testing.
    ds_train = Dataset(df_train, label=target_column, cat_features=cat_columns)
    ds_test = Dataset(df_test, label=target_column, cat_features=cat_columns)

    # Prepare full suite testing object.
    suite = full_suite()

    # Run Tests.
    suite_result = suite.run(train_dataset=ds_train, test_dataset=ds_test, model=model)

    suite_result.show_in_iframe()
