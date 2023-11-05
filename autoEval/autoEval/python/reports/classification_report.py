import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


def __plot_confusion_matrix_thresholds(y_true, y_prob, datatype):
    thresholds = [0.3, 0.5, 0.7]
    class_names = y_true.unique().tolist()
    for threshold in thresholds:
        y_prob_np = np.array(y_prob)  # Convert the list to a NumPy array
        y_prob_positive = y_prob_np[
            :, 1
        ]  # Assuming the second column represents the positive class probabilities
        y_pred = (y_prob_positive > threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure()
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion matrix {datatype}" + " (Threshold = %0.2f)" % threshold)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        fmt = "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()


def visualize_classification_predictions(
    train_predictions, test_predictions, target, prediction_column
):
    # Plot Confusion Matrix based on threshhold.
    __plot_confusion_matrix_thresholds(
        train_predictions[target],
        train_predictions[f"{prediction_column}_Probability"].tolist(),
        "Train",
    )

    __plot_confusion_matrix_thresholds(
        test_predictions[target],
        test_predictions[f"{prediction_column}_Probability"].tolist(),
        "Test",
    )
