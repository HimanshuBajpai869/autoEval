import matplotlib.pyplot as plt


def __plot_residual_frequency_plot(
    train_predictions,
    test_predictions,
    target,
    prediction_column,
):
    train_residuals = train_predictions[target] - train_predictions[prediction_column]

    plt.hist(train_residuals, bins=10, color="skyblue", edgecolor="black")
    plt.title("Frequency Plot of Residual Errors for Train Predictions")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()

    test_residuals = test_predictions[target] - test_predictions[prediction_column]

    plt.hist(test_residuals, bins=10, color="grey", edgecolor="black")
    plt.title("Frequency Plot of Residual Errors for Test Predictions")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.show()


def __plot_actual_vs_predicted_scatter_plot(
    train_predictions,
    test_predictions,
    target,
    prediction_column,
):
    plt.scatter(
        train_predictions[target], train_predictions[prediction_column], c="skyblue"
    )
    plt.title("Actual vs Predicted Plot for Train Predictions")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.show()

    plt.scatter(test_predictions[target], test_predictions[prediction_column], c="grey")
    plt.title("Actual vs Predicted Plot for Test Predictions")
    plt.xlabel("Actual Value")
    plt.ylabel("Predicted Value")
    plt.show()


def visualize_regression_predictions(
    train_predictions, test_predictions, target, prediction_column
):
    # Plot Residual Frequency Plot
    __plot_residual_frequency_plot(
        train_predictions,
        test_predictions,
        target,
        prediction_column,
    )

    # Plot Actual vs Predicted Scatter Plot
    __plot_actual_vs_predicted_scatter_plot(
        train_predictions,
        test_predictions,
        target,
        prediction_column,
    )
