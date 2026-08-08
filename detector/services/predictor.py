from functools import lru_cache
from io import BytesIO
from threading import Lock
import json

import numpy as np

from ai_edge_litert.interpreter import Interpreter
from django.conf import settings
from PIL import Image, ImageOps

# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = (
    settings.BASE_DIR / "model" / "vgg19_ulcer_disease_finetuned_optimized.tflite"
)

CLASS_NAMES_PATH = settings.BASE_DIR / "model" / "class_names.json"


# ============================================================
# DISPLAY LABELS
# ============================================================

DISPLAY_NAMES = {
    "normal": "Normal",
    "esophagitis": "Esophagitis",
    "ulcerative-colitis": "Ulcerative Colitis",
    "polyps": "Polyps",
}


# ============================================================
# THREAD SAFETY
# ============================================================

# One LiteRT interpreter is reused by Django.
#
# The lock prevents two requests from changing interpreter
# tensors at the same time.

INTERPRETER_LOCK = Lock()


# ============================================================
# LOAD CLASS CONFIGURATION
# ============================================================


@lru_cache(maxsize=1)
def load_class_configuration():
    """
    Load class names and image size once.
    """

    if not CLASS_NAMES_PATH.exists():

        raise FileNotFoundError(
            "Class configuration was not found: " f"{CLASS_NAMES_PATH}"
        )

    with CLASS_NAMES_PATH.open(
        "r",
        encoding="utf-8",
    ) as configuration_file:

        configuration = json.load(configuration_file)

    class_names = configuration["class_names"]

    image_size = tuple(configuration["image_size"])

    return (
        class_names,
        image_size,
    )


# ============================================================
# LOAD LITERT MODEL
# ============================================================


@lru_cache(maxsize=1)
def load_interpreter():
    """
    Load the optimized LiteRT model once.

    Reusing one interpreter prevents the model from
    being repeatedly loaded for every prediction.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "LiteRT prediction model was not found: " f"{MODEL_PATH}"
        )

    interpreter = Interpreter(
        model_path=str(MODEL_PATH),
        num_threads=2,
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]

    output_details = interpreter.get_output_details()[0]

    return (
        interpreter,
        input_details,
        output_details,
    )


# ============================================================
# IMAGE PREPROCESSING
# ============================================================


def prepare_image(
    image_bytes: bytes,
    image_size: tuple[int, int],
    input_dtype,
) -> np.ndarray:
    """
    Prepare one uploaded image for LiteRT inference.

    The converted model already contains the VGG19-specific
    preprocessing operations.

    Therefore this function only:

    1. Opens the uploaded image.
    2. Corrects EXIF orientation.
    3. Converts the image to RGB.
    4. Resizes it to 112 x 112.
    5. Converts it to the model input dtype.
    6. Adds the batch dimension.
    """

    image_height, image_width = image_size

    with Image.open(BytesIO(image_bytes)) as image:

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
# RUN LITERT INFERENCE
# ============================================================


def run_inference(
    image_batch: np.ndarray,
) -> np.ndarray:
    """
    Run one prediction through the LiteRT interpreter.
    """

    (
        interpreter,
        input_details,
        output_details,
    ) = load_interpreter()

    with INTERPRETER_LOCK:

        interpreter.set_tensor(
            input_details["index"],
            image_batch,
        )

        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details["index"])[0]

    return predictions


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================


def predict_endoscopy_image(
    image_bytes: bytes,
) -> dict:
    """
    Analyze an uploaded gastrointestinal endoscopy image.

    Returns:
        predicted class,
        display label,
        confidence percentage,
        probability for each supported class.
    """

    (
        class_names,
        image_size,
    ) = load_class_configuration()

    (
        _,
        input_details,
        _,
    ) = load_interpreter()

    image_batch = prepare_image(
        image_bytes=image_bytes,
        image_size=image_size,
        input_dtype=input_details["dtype"],
    )

    predictions = run_inference(image_batch)

    predicted_index = int(np.argmax(predictions))

    predicted_class = class_names[predicted_index]

    confidence = float(predictions[predicted_index])

    probabilities = []

    for index, class_name in enumerate(class_names):

        probability = float(predictions[index])

        probabilities.append(
            {
                "name": class_name,
                "label": DISPLAY_NAMES.get(
                    class_name,
                    class_name.replace(
                        "-",
                        " ",
                    ).title(),
                ),
                "value": round(
                    probability * 100,
                    2,
                ),
            }
        )

    # Highest probability appears first in the UI.
    probabilities.sort(
        key=lambda item: item["value"],
        reverse=True,
    )

    return {
        "prediction": predicted_class,
        "prediction_label": DISPLAY_NAMES.get(
            predicted_class,
            predicted_class.replace(
                "-",
                " ",
            ).title(),
        ),
        "confidence": round(
            confidence * 100,
            2,
        ),
        "probabilities": probabilities,
    }
