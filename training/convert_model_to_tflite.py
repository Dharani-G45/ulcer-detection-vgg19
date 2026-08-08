from pathlib import Path
import os

# Reduce TensorFlow informational output.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIRECTORY = PROJECT_ROOT / "model"


SOURCE_MODEL_PATH = MODEL_DIRECTORY / "vgg19_ulcer_disease_finetuned.keras"


FLOAT32_TFLITE_PATH = MODEL_DIRECTORY / "vgg19_ulcer_disease_finetuned_float32.tflite"


OPTIMIZED_TFLITE_PATH = (
    MODEL_DIRECTORY / "vgg19_ulcer_disease_finetuned_optimized.tflite"
)


# ============================================================
# HELPER
# ============================================================


def file_size_mb(
    file_path: Path,
) -> float:
    """
    Return file size in megabytes.
    """

    return file_path.stat().st_size / (1024 * 1024)


# ============================================================
# CHECK SOURCE MODEL
# ============================================================


def check_source_model() -> None:
    """
    Ensure the fine-tuned Keras model exists.
    """

    if not SOURCE_MODEL_PATH.exists():

        raise FileNotFoundError(
            "\nFine-tuned model was not found:\n" f"{SOURCE_MODEL_PATH}"
        )


# ============================================================
# LOAD KERAS MODEL
# ============================================================


def load_keras_model():
    """
    Load the fine-tuned VGG19 model.
    """

    print("\nLoading fine-tuned Keras model...")

    model = tf.keras.models.load_model(
        SOURCE_MODEL_PATH,
        compile=False,
    )

    print("Model loaded successfully.")

    print(
        "\nInput shape:",
        model.input_shape,
    )

    print(
        "Output shape:",
        model.output_shape,
    )

    return model


# ============================================================
# FLOAT32 CONVERSION
# ============================================================


def convert_float32_model(
    model,
) -> None:
    """
    Convert Keras model to standard Float32 TFLite.
    """

    print("\n" "==============================================")

    print("CONVERTING FLOAT32 TFLITE MODEL")

    print("==============================================")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Require normal built-in TFLite operations.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tflite_model = converter.convert()

    FLOAT32_TFLITE_PATH.write_bytes(tflite_model)

    print("\nFloat32 model created:")

    print(FLOAT32_TFLITE_PATH)

    print(f"\nSize: " f"{file_size_mb(FLOAT32_TFLITE_PATH):.2f} MB")


# ============================================================
# OPTIMIZED CONVERSION
# ============================================================


def convert_optimized_model(
    model,
) -> None:
    """
    Convert Keras model using post-training optimization.

    Optimize.DEFAULT allows TensorFlow Lite to reduce
    model size and improve deployment efficiency.
    """

    print("\n" "==============================================")

    print("CONVERTING OPTIMIZED TFLITE MODEL")

    print("==============================================")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    optimized_model = converter.convert()

    OPTIMIZED_TFLITE_PATH.write_bytes(optimized_model)

    print("\nOptimized model created:")

    print(OPTIMIZED_TFLITE_PATH)

    print(f"\nSize: " f"{file_size_mb(OPTIMIZED_TFLITE_PATH):.2f} MB")


# ============================================================
# TEST TFLITE MODEL
# ============================================================


def validate_tflite_model(
    model_path: Path,
) -> None:
    """
    Load the TFLite model and perform one test inference.

    This verifies that the converted file can actually
    be executed before deployment.
    """

    print("\n" "----------------------------------------------")

    print(f"Validating:\n{model_path.name}")

    print("----------------------------------------------")

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=1,
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    output_details = interpreter.get_output_details()

    input_info = input_details[0]

    output_info = output_details[0]

    print(
        "\nInput shape:",
        input_info["shape"],
    )

    print(
        "Input dtype:",
        input_info["dtype"],
    )

    print(
        "Output shape:",
        output_info["shape"],
    )

    print(
        "Output dtype:",
        output_info["dtype"],
    )

    input_shape = tuple(input_info["shape"])

    input_dtype = input_info["dtype"]

    # Create a dummy image.
    dummy_input = np.zeros(
        input_shape,
        dtype=input_dtype,
    )

    interpreter.set_tensor(
        input_info["index"],
        dummy_input,
    )

    interpreter.invoke()

    output = interpreter.get_tensor(output_info["index"])

    print("\nTest inference output:")

    print(output)

    print("\nOutput probability sum:")

    print(float(np.sum(output)))

    print("\nTFLite model executed successfully.")


# ============================================================
# COMPARE FILE SIZES
# ============================================================


def print_size_summary() -> None:
    """
    Compare Keras and TFLite model sizes.
    """

    keras_size = file_size_mb(SOURCE_MODEL_PATH)

    float32_size = file_size_mb(FLOAT32_TFLITE_PATH)

    optimized_size = file_size_mb(OPTIMIZED_TFLITE_PATH)

    reduction = (keras_size - optimized_size) / keras_size * 100

    print("\n" "==============================================")

    print("MODEL SIZE COMPARISON")

    print("==============================================")

    print(f"Keras model:     " f"{keras_size:.2f} MB")

    print(f"Float32 TFLite:  " f"{float32_size:.2f} MB")

    print(f"Optimized TFLite:" f" {optimized_size:.2f} MB")

    print("----------------------------------------------")

    print(f"Optimized size reduction: " f"{reduction:.2f}%")

    print("==============================================")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """
    Complete Keras-to-TFLite conversion workflow.
    """

    print("\n" "==============================================")

    print("ULCER DETECTION - TFLITE CONVERSION")

    print("==============================================")

    print(f"\nTensorFlow version: " f"{tf.__version__}")

    check_source_model()

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_keras_model()

    convert_float32_model(model)

    validate_tflite_model(FLOAT32_TFLITE_PATH)

    convert_optimized_model(model)

    validate_tflite_model(OPTIMIZED_TFLITE_PATH)

    print_size_summary()

    print("\n" "==============================================")

    print("CONVERSION COMPLETED SUCCESSFULLY")

    print("==============================================")

    print("\nNext step:")

    print("Evaluate the optimized TFLite model " "using all 600 test images.")


if __name__ == "__main__":
    main()
