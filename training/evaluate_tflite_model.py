from pathlib import Path
import json
import os

# Reduce unnecessary TensorFlow logs.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps

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

TEST_DIRECTORY = PROJECT_ROOT / "dataset_split" / "test"


TFLITE_MODEL_PATH = MODEL_DIRECTORY / "vgg19_ulcer_disease_finetuned_optimized.tflite"


CLASS_NAMES_PATH = MODEL_DIRECTORY / "class_names.json"


# ============================================================
# OUTPUT PATHS
# ============================================================

EVALUATION_DIRECTORY = MODEL_DIRECTORY / "evaluation_tflite"


METRICS_PATH = EVALUATION_DIRECTORY / "test_metrics.json"


CLASSIFICATION_REPORT_PATH = EVALUATION_DIRECTORY / "classification_report.txt"


CONFUSION_MATRIX_IMAGE_PATH = EVALUATION_DIRECTORY / "confusion_matrix.png"


CONFUSION_MATRIX_JSON_PATH = EVALUATION_DIRECTORY / "confusion_matrix.json"


# ============================================================
# SUPPORTED IMAGE TYPES
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# CHECK REQUIRED FILES
# ============================================================


def check_required_files():
    """
    Confirm that the optimized model, class mapping
    and test dataset exist.
    """

    required_paths = [
        TFLITE_MODEL_PATH,
        CLASS_NAMES_PATH,
        TEST_DIRECTORY,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                "\nRequired file or folder " f"was not found:\n{path}"
            )


# ============================================================
# LOAD MODEL CONFIGURATION
# ============================================================


def load_configuration():
    """
    Load class order and image size saved during training.
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
# LOAD TFLITE MODEL
# ============================================================


def load_tflite_interpreter():
    """
    Create the TensorFlow Lite interpreter.
    """

    print("\nLoading optimized TFLite model...")

    interpreter = tf.lite.Interpreter(
        model_path=str(TFLITE_MODEL_PATH),
        num_threads=2,
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]

    output_details = interpreter.get_output_details()[0]

    print("TFLite model loaded successfully.")

    print(
        "\nInput shape:",
        input_details["shape"],
    )

    print(
        "Input dtype:",
        input_details["dtype"],
    )

    print(
        "Output shape:",
        output_details["shape"],
    )

    print(
        "Output dtype:",
        output_details["dtype"],
    )

    return (
        interpreter,
        input_details,
        output_details,
    )


# ============================================================
# GET TEST IMAGES
# ============================================================


def collect_test_images(
    class_names,
):
    """
    Collect all 600 test images while preserving
    the configured class order.
    """

    image_records = []

    for class_index, class_name in enumerate(class_names):

        class_directory = TEST_DIRECTORY / class_name

        if not class_directory.exists():

            raise FileNotFoundError(
                "\nTest class directory " f"was not found:\n{class_directory}"
            )

        image_files = sorted(
            [
                file
                for file in class_directory.iterdir()
                if (file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS)
            ]
        )

        print(f"{class_name:<20}" f"{len(image_files):>4} images")

        for image_path in image_files:

            image_records.append(
                {
                    "path": image_path,
                    "label": class_index,
                    "class_name": class_name,
                }
            )

    return image_records


# ============================================================
# PREPARE IMAGE
# ============================================================


def prepare_image(
    image_path: Path,
    image_size,
    input_dtype,
):
    """
    Prepare one image exactly as the Django application will.

    Important:
    The converted model already contains the VGG19
    preprocessing operations.

    Therefore we only:
    - load image
    - fix orientation
    - convert RGB
    - resize
    - convert to float32
    - add batch dimension
    """

    image_height, image_width = image_size

    with Image.open(image_path) as image:

        image = ImageOps.exif_transpose(image)

        image = image.convert("RGB")

        image = image.resize(
            (
                image_width,
                image_height,
            ),
            Image.Resampling.LANCZOS,
        )

        image_array = np.asarray(
            image,
            dtype=input_dtype,
        )

    image_batch = np.expand_dims(
        image_array,
        axis=0,
    )

    return image_batch


# ============================================================
# RUN ONE PREDICTION
# ============================================================


def predict_image(
    interpreter,
    input_details,
    output_details,
    image_batch,
):
    """
    Run inference for one test image.
    """

    interpreter.set_tensor(
        input_details["index"],
        image_batch,
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details["index"])[0]

    predicted_index = int(np.argmax(predictions))

    return (
        predictions,
        predicted_index,
    )


# ============================================================
# EVALUATE ALL TEST IMAGES
# ============================================================


def evaluate_all_images(
    interpreter,
    input_details,
    output_details,
    image_records,
    image_size,
):
    """
    Evaluate all 600 untouched test images.
    """

    true_labels = []

    predicted_labels = []

    total_images = len(image_records)

    print(f"\nEvaluating " f"{total_images} test images...")

    for index, record in enumerate(
        image_records,
        start=1,
    ):

        image_batch = prepare_image(
            image_path=record["path"],
            image_size=image_size,
            input_dtype=input_details["dtype"],
        )

        (
            _,
            predicted_index,
        ) = predict_image(
            interpreter=interpreter,
            input_details=input_details,
            output_details=output_details,
            image_batch=image_batch,
        )

        true_labels.append(record["label"])

        predicted_labels.append(predicted_index)

        if index % 50 == 0 or index == total_images:

            print(f"Processed " f"{index}/{total_images}")

    return (
        np.array(true_labels),
        np.array(predicted_labels),
    )


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
    Create and save the optimized model confusion matrix.
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
        title=("Optimized TFLite " "Test Confusion Matrix"),
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
    Calculate accuracy separately for each class.
    """

    results = {}

    for index, class_name in enumerate(class_names):

        total = matrix[index].sum()

        correct = matrix[
            index,
            index,
        ]

        accuracy = correct / total if total > 0 else 0.0

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
    metrics,
    class_results,
    matrix,
):
    """
    Save optimized TFLite evaluation results.
    """

    result_data = {
        "model": ("vgg19_ulcer_disease_" "finetuned_optimized.tflite"),
        "test_accuracy": float(metrics["accuracy"]),
        "weighted_precision": float(metrics["precision"]),
        "weighted_recall": float(metrics["recall"]),
        "weighted_f1_score": float(metrics["f1_score"]),
        "per_class": (class_results),
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
    metrics,
    class_results,
):
    """
    Display optimized TFLite test results.
    """

    print("\n" "================================================")

    print("OPTIMIZED TFLITE TEST RESULTS")

    print("================================================")

    print(f"\nTest Accuracy: " f"{metrics['accuracy'] * 100:.2f}%")

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
    Complete optimized TFLite evaluation workflow.
    """

    print("\n" "================================================")

    print("ULCER DETECTION - TFLITE MODEL EVALUATION")

    print("================================================")

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

    print("\nTest dataset:")

    image_records = collect_test_images(class_names)

    (
        interpreter,
        input_details,
        output_details,
    ) = load_tflite_interpreter()

    (
        true_labels,
        predicted_labels,
    ) = evaluate_all_images(
        interpreter=interpreter,
        input_details=input_details,
        output_details=output_details,
        image_records=image_records,
        image_size=image_size,
    )

    metrics = calculate_metrics(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=class_names,
    )

    matrix = create_confusion_matrix(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=class_names,
    )

    class_results = calculate_per_class_accuracy(
        matrix=matrix,
        class_names=class_names,
    )

    save_results(
        metrics=metrics,
        class_results=class_results,
        matrix=matrix,
    )

    print_results(
        metrics=metrics,
        class_results=class_results,
    )

    print("\nEvaluation files saved to:")

    print(EVALUATION_DIRECTORY)

    print("\nOptimized TFLite " "evaluation completed successfully.")


if __name__ == "__main__":
    main()
