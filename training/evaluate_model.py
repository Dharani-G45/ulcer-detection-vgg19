from pathlib import Path
import json
import os

# Reduce TensorFlow information messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

# Allows graphs to be saved without opening a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIRECTORY = PROJECT_ROOT / "model"

MODEL_PATH = MODEL_DIRECTORY / "vgg19_ulcer_disease.keras"

CLASS_NAMES_PATH = MODEL_DIRECTORY / "class_names.json"

TEST_DIRECTORY = PROJECT_ROOT / "dataset_split" / "test"


# ============================================================
# EVALUATION OUTPUT
# ============================================================

EVALUATION_DIRECTORY = MODEL_DIRECTORY / "evaluation"

METRICS_PATH = EVALUATION_DIRECTORY / "test_metrics.json"

CLASSIFICATION_REPORT_PATH = EVALUATION_DIRECTORY / "classification_report.txt"

CONFUSION_MATRIX_PATH = EVALUATION_DIRECTORY / "confusion_matrix.png"

CONFUSION_MATRIX_DATA_PATH = EVALUATION_DIRECTORY / "confusion_matrix.json"


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16


# ============================================================
# VALIDATION
# ============================================================


def check_required_files() -> None:
    """
    Confirm that all required model and dataset files exist.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(f"\nModel was not found:\n" f"{MODEL_PATH}")

    if not CLASS_NAMES_PATH.exists():

        raise FileNotFoundError(
            f"\nClass mapping was not found:\n" f"{CLASS_NAMES_PATH}"
        )

    if not TEST_DIRECTORY.exists():

        raise FileNotFoundError(f"\nTest dataset was not found:\n" f"{TEST_DIRECTORY}")


# ============================================================
# LOAD CLASS CONFIGURATION
# ============================================================


def load_class_configuration():
    """
    Read the class order and image size saved during training.
    """

    with CLASS_NAMES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        configuration = json.load(file)

    class_names = configuration["class_names"]

    image_size = tuple(configuration["image_size"])

    return (
        class_names,
        image_size,
    )


# ============================================================
# LOAD TEST DATASET
# ============================================================


def load_test_dataset(
    class_names,
    image_size,
):
    """
    Load the untouched test dataset.

    shuffle=False is essential because predictions and
    ground-truth labels must remain in the same order.
    """

    print("\nLoading test dataset...")

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIRECTORY,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=image_size,
        shuffle=False,
    )

    return test_dataset.prefetch(tf.data.AUTOTUNE)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================


def load_trained_model():
    """
    Load the best trained VGG19 model.
    """

    print("\nLoading trained VGG19 model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    return model


# ============================================================
# GET TRUE LABELS
# ============================================================


def get_true_labels(
    test_dataset,
):
    """
    Convert one-hot encoded labels into integer class IDs.
    """

    true_labels = []

    for _, labels in test_dataset:

        batch_labels = np.argmax(
            labels.numpy(),
            axis=1,
        )

        true_labels.extend(batch_labels.tolist())

    return np.array(true_labels)


# ============================================================
# MODEL PREDICTIONS
# ============================================================


def get_predictions(
    model,
    test_dataset,
):
    """
    Run predictions for all unseen test images.
    """

    print("\nRunning predictions on " "the test dataset...")

    prediction_probabilities = model.predict(
        test_dataset,
        verbose=1,
    )

    predicted_labels = np.argmax(
        prediction_probabilities,
        axis=1,
    )

    return (
        prediction_probabilities,
        predicted_labels,
    )


# ============================================================
# MODEL EVALUATION
# ============================================================


def evaluate_model(
    model,
    test_dataset,
):
    """
    Calculate TensorFlow test loss and accuracy.
    """

    print("\nEvaluating model...")

    evaluation_result = model.evaluate(
        test_dataset,
        verbose=1,
        return_dict=True,
    )

    return evaluation_result


# ============================================================
# CLASSIFICATION METRICS
# ============================================================


def calculate_metrics(
    true_labels,
    predicted_labels,
    class_names,
):
    """
    Calculate overall precision, recall, F1-score and accuracy.
    """

    accuracy = accuracy_score(
        true_labels,
        predicted_labels,
    )

    (
        precision,
        recall,
        f1_score,
        _,
    ) = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "classification_report": report,
    }


# ============================================================
# CONFUSION MATRIX
# ============================================================


def create_confusion_matrix(
    true_labels,
    predicted_labels,
    class_names,
):
    """
    Generate and save a confusion matrix.
    """

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    figure, axis = plt.subplots(figsize=(9, 7))

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=("VGG19 Test Dataset " "Confusion Matrix"),
        ylabel="Actual Class",
        xlabel="Predicted Class",
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(matrix.shape[0]):

        for column in range(matrix.shape[1]):

            value = matrix[row, column]

            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color=("white" if value > threshold else "black"),
                fontsize=11,
                fontweight="bold",
            )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)

    return matrix


# ============================================================
# PER-CLASS ACCURACY
# ============================================================


def calculate_per_class_accuracy(
    matrix,
    class_names,
):
    """
    Calculate classification accuracy separately for each class.
    """

    class_results = {}

    for index, class_name in enumerate(class_names):

        total_images = matrix[index].sum()

        correct_predictions = matrix[
            index,
            index,
        ]

        if total_images > 0:

            accuracy = correct_predictions / total_images

        else:

            accuracy = 0.0

        class_results[class_name] = {
            "correct": int(correct_predictions),
            "total": int(total_images),
            "accuracy": float(accuracy),
        }

    return class_results


# ============================================================
# SAVE RESULTS
# ============================================================


def save_results(
    tensorflow_result,
    calculated_metrics,
    class_results,
    confusion_matrix_data,
):
    """
    Save all test results to files.
    """

    results = {
        "test_loss": float(tensorflow_result.get("loss", 0)),
        "test_accuracy": float(calculated_metrics["accuracy"]),
        "weighted_precision": float(calculated_metrics["precision"]),
        "weighted_recall": float(calculated_metrics["recall"]),
        "weighted_f1_score": float(calculated_metrics["f1_score"]),
        "per_class": class_results,
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    with CLASSIFICATION_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(calculated_metrics["classification_report"])

    with CONFUSION_MATRIX_DATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            confusion_matrix_data.tolist(),
            file,
            indent=4,
        )


# ============================================================
# DISPLAY RESULTS
# ============================================================


def print_results(
    tensorflow_result,
    metrics,
    class_results,
):
    """
    Print evaluation results to the terminal.
    """

    print("\n" "================================================")

    print("VGG19 TEST DATASET RESULTS")

    print("================================================")

    print(f"\nTest Loss: " f"{tensorflow_result.get('loss', 0):.4f}")

    print(f"Test Accuracy: " f"{metrics['accuracy'] * 100:.2f}%")

    print(f"Weighted Precision: " f"{metrics['precision'] * 100:.2f}%")

    print(f"Weighted Recall: " f"{metrics['recall'] * 100:.2f}%")

    print(f"Weighted F1-Score: " f"{metrics['f1_score'] * 100:.2f}%")

    print("\n" "------------------------------------------------")

    print("PER-CLASS ACCURACY")

    print("------------------------------------------------")

    for class_name, result in class_results.items():

        print(
            f"{class_name:<20}"
            f"{result['correct']:>3}"
            f"/{result['total']:<3}"
            f"  "
            f"{result['accuracy'] * 100:>6.2f}%"
        )

    print("\n" "------------------------------------------------")

    print("CLASSIFICATION REPORT")

    print("------------------------------------------------")

    print(metrics["classification_report"])

    print("================================================")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """
    Complete VGG19 test evaluation workflow.
    """

    print("\n" "================================================")

    print("ULCER DETECTION - MODEL EVALUATION")

    print("================================================")

    print(f"\nTensorFlow: " f"{tf.__version__}")

    check_required_files()

    EVALUATION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        class_names,
        image_size,
    ) = load_class_configuration()

    print("\nClass order:")

    for index, class_name in enumerate(class_names):

        print(f"  {index} -> {class_name}")

    print(f"\nImage size: " f"{image_size[0]} x " f"{image_size[1]}")

    test_dataset = load_test_dataset(
        class_names,
        image_size,
    )

    model = load_trained_model()

    tensorflow_result = evaluate_model(
        model,
        test_dataset,
    )

    true_labels = get_true_labels(test_dataset)

    (
        _,
        predicted_labels,
    ) = get_predictions(
        model,
        test_dataset,
    )

    metrics = calculate_metrics(
        true_labels,
        predicted_labels,
        class_names,
    )

    matrix = create_confusion_matrix(
        true_labels,
        predicted_labels,
        class_names,
    )

    class_results = calculate_per_class_accuracy(
        matrix,
        class_names,
    )

    save_results(
        tensorflow_result,
        metrics,
        class_results,
        matrix,
    )

    print_results(
        tensorflow_result,
        metrics,
        class_results,
    )

    print("\nEvaluation files saved to:")

    print(EVALUATION_DIRECTORY)

    print("\nModel evaluation completed successfully.")


if __name__ == "__main__":
    main()
