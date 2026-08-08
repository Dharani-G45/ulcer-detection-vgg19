from pathlib import Path
import json
import os

# Reduce TensorFlow informational logs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

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

MODEL_PATH = MODEL_DIRECTORY / "vgg19_ulcer_disease_finetuned.keras"

CLASS_NAMES_PATH = MODEL_DIRECTORY / "class_names.json"

TEST_DIRECTORY = PROJECT_ROOT / "dataset_split" / "test"


# ============================================================
# OUTPUT PATHS
# ============================================================

EVALUATION_DIRECTORY = MODEL_DIRECTORY / "evaluation_finetuned"

METRICS_PATH = EVALUATION_DIRECTORY / "test_metrics.json"

CLASSIFICATION_REPORT_PATH = EVALUATION_DIRECTORY / "classification_report.txt"

CONFUSION_MATRIX_IMAGE_PATH = EVALUATION_DIRECTORY / "confusion_matrix.png"

CONFUSION_MATRIX_JSON_PATH = EVALUATION_DIRECTORY / "confusion_matrix.json"


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16


# ============================================================
# CHECK REQUIRED FILES
# ============================================================


def check_required_files():
    """
    Verify that all required files and directories exist.
    """

    required_paths = [
        MODEL_PATH,
        CLASS_NAMES_PATH,
        TEST_DIRECTORY,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(f"\nRequired file/folder not found:\n" f"{path}")


# ============================================================
# LOAD CONFIGURATION
# ============================================================


def load_configuration():
    """
    Load class names and image size saved during training.
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
    """

    print("\nLoading untouched test dataset...")

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

    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return test_dataset


# ============================================================
# LOAD FINE-TUNED MODEL
# ============================================================


def load_model():
    """
    Load the fine-tuned VGG19 model.
    """

    print("\nLoading fine-tuned VGG19 model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Fine-tuned model loaded successfully.")

    return model


# ============================================================
# TRUE LABELS
# ============================================================


def get_true_labels(
    dataset,
):
    """
    Extract integer class labels from one-hot labels.
    """

    labels = []

    for _, batch_labels in dataset:

        batch_indices = np.argmax(
            batch_labels.numpy(),
            axis=1,
        )

        labels.extend(batch_indices.tolist())

    return np.array(labels)


# ============================================================
# PREDICTIONS
# ============================================================


def make_predictions(
    model,
    dataset,
):
    """
    Generate model predictions for all test images.
    """

    print("\nRunning fine-tuned model predictions...")

    probabilities = model.predict(
        dataset,
        verbose=1,
    )

    predicted_labels = np.argmax(
        probabilities,
        axis=1,
    )

    return (
        probabilities,
        predicted_labels,
    )


# ============================================================
# TENSORFLOW EVALUATION
# ============================================================


def evaluate_tensorflow_model(
    model,
    dataset,
):
    """
    Calculate TensorFlow loss and accuracy.
    """

    print("\nEvaluating fine-tuned model...")

    results = model.evaluate(
        dataset,
        verbose=1,
        return_dict=True,
    )

    return results


# ============================================================
# CALCULATE METRICS
# ============================================================


def calculate_metrics(
    true_labels,
    predicted_labels,
    class_names,
):
    """
    Calculate accuracy, precision, recall and F1-score.
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
    Create and save confusion matrix.
    """

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    figure, axis = plt.subplots(figsize=(9, 7))

    image = axis.imshow(
        matrix,
        cmap="Greens",
        interpolation="nearest",
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
        xlabel="Predicted Class",
        ylabel="Actual Class",
        title=("Fine-Tuned VGG19 " "Test Confusion Matrix"),
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
        CONFUSION_MATRIX_IMAGE_PATH,
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
    Calculate accuracy for every individual class.
    """

    results = {}

    for index, class_name in enumerate(class_names):

        total = matrix[index].sum()

        correct = matrix[
            index,
            index,
        ]

        if total > 0:

            accuracy = correct / total

        else:

            accuracy = 0.0

        results[class_name] = {
            "correct": int(correct),
            "total": int(total),
            "accuracy": float(accuracy),
        }

    return results


# ============================================================
# SAVE RESULTS
# ============================================================


def save_results(
    tensorflow_results,
    metrics,
    class_results,
    matrix,
):
    """
    Save evaluation information.
    """

    result_data = {
        "model": ("vgg19_ulcer_disease_finetuned.keras"),
        "test_loss": float(tensorflow_results.get("loss", 0)),
        "test_accuracy": float(metrics["accuracy"]),
        "weighted_precision": float(metrics["precision"]),
        "weighted_recall": float(metrics["recall"]),
        "weighted_f1_score": float(metrics["f1_score"]),
        "per_class": class_results,
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result_data,
            file,
            indent=4,
        )

    with CLASSIFICATION_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(metrics["classification_report"])

    with CONFUSION_MATRIX_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            matrix.tolist(),
            file,
            indent=4,
        )


# ============================================================
# PRINT RESULTS
# ============================================================


def print_results(
    tensorflow_results,
    metrics,
    class_results,
):
    """
    Display final test results.
    """

    print("\n" "================================================")

    print("FINE-TUNED VGG19 TEST RESULTS")

    print("================================================")

    print(f"\nTest Loss: " f"{tensorflow_results.get('loss', 0):.4f}")

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


def main():
    """
    Complete fine-tuned model evaluation workflow.
    """

    print("\n" "================================================")

    print("ULCER DETECTION - FINE-TUNED MODEL EVALUATION")

    print("================================================")

    print(f"\nTensorFlow version: " f"{tf.__version__}")

    check_required_files()

    EVALUATION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        class_names,
        image_size,
    ) = load_configuration()

    print("\nClass order:")

    for index, class_name in enumerate(class_names):

        print(f"  {index} -> {class_name}")

    print(f"\nImage size: " f"{image_size[0]} x " f"{image_size[1]}")

    dataset = load_test_dataset(
        class_names,
        image_size,
    )

    model = load_model()

    tensorflow_results = evaluate_tensorflow_model(
        model,
        dataset,
    )

    true_labels = get_true_labels(dataset)

    (
        _,
        predicted_labels,
    ) = make_predictions(
        model,
        dataset,
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
        tensorflow_results,
        metrics,
        class_results,
        matrix,
    )

    print_results(
        tensorflow_results,
        metrics,
        class_results,
    )

    print("\nEvaluation files saved to:")

    print(EVALUATION_DIRECTORY)

    print("\nFine-tuned model evaluation completed.")


if __name__ == "__main__":
    main()
