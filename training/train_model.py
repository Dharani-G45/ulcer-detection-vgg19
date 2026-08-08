from pathlib import Path
import json
import os

# Reduce unnecessary TensorFlow INFO messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

# Allows plots to be saved without opening a GUI window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Input,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomZoom,
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset_split"

TRAIN_DIRECTORY = DATASET_ROOT / "train"
VALIDATION_DIRECTORY = DATASET_ROOT / "validation"

MODEL_DIRECTORY = PROJECT_ROOT / "model"

MODEL_PATH = MODEL_DIRECTORY / "vgg19_ulcer_disease.keras"

LABELS_PATH = MODEL_DIRECTORY / "class_names.json"

HISTORY_IMAGE_PATH = MODEL_DIRECTORY / "training_history.png"

TRAINING_LOG_PATH = MODEL_DIRECTORY / "training_log.csv"

MODEL_SUMMARY_PATH = MODEL_DIRECTORY / "model_summary.txt"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMAGE_HEIGHT = 112
IMAGE_WIDTH = 112

IMAGE_SIZE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)

INPUT_SHAPE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    3,
)


BATCH_SIZE = 16

EPOCHS = 10

LEARNING_RATE = 0.0001

RANDOM_SEED = 42


# IMPORTANT:
#
# The order of these classes determines the numeric output
# produced by the neural network.
#
# 0 -> normal
# 1 -> esophagitis
# 2 -> ulcerative-colitis
# 3 -> polyps

CLASS_NAMES = [
    "normal",
    "esophagitis",
    "ulcerative-colitis",
    "polyps",
]


# ============================================================
# DATASET VALIDATION
# ============================================================


def check_required_directories() -> None:
    """
    Ensure the train and validation directories exist.
    """

    required_directories = [
        TRAIN_DIRECTORY,
        VALIDATION_DIRECTORY,
    ]

    for directory in required_directories:

        if not directory.exists():

            raise FileNotFoundError(
                "\nRequired dataset directory " f"was not found:\n{directory}"
            )


# ============================================================
# LOAD DATASETS
# ============================================================


def load_datasets():
    """
    Load training and validation datasets.

    Class order is explicitly supplied so TensorFlow does not
    assign classes alphabetically.
    """

    print("\nLoading training dataset...")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIRECTORY,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    print("\nLoading validation dataset...")

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIRECTORY,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False,
    )

    print("\nClass order:")

    for index, class_name in enumerate(CLASS_NAMES):

        print(f"  {index} -> {class_name}")

    return (
        train_dataset,
        validation_dataset,
    )
# ============================================================
# DATA PIPELINE
# ============================================================


def optimize_datasets(
    train_dataset,
    validation_dataset,
):
    """
    Prefetch data while the model is training.

    This lets TensorFlow prepare upcoming batches while the
    current batch is being processed.
    """

    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=autotune)

    validation_dataset = validation_dataset.prefetch(buffer_size=autotune)

    return (
        train_dataset,
        validation_dataset,
    )


# ============================================================
# DATA AUGMENTATION
# ============================================================


def create_data_augmentation():
    """
    Create mild image augmentation.

    We deliberately avoid very aggressive transformations
    because these are medical endoscopy images.
    """

    augmentation = tf.keras.Sequential(
        [
            RandomFlip(mode="horizontal"),
            RandomRotation(factor=0.05),
            RandomZoom(
                height_factor=0.10,
                width_factor=0.10,
            ),
            RandomContrast(factor=0.10),
        ],
        name="data_augmentation",
    )

    return augmentation


# ============================================================
# BUILD MODEL
# ============================================================


def build_model() -> Model:
    """
    Create the VGG19 transfer-learning model.
    """

    print("\nLoading pretrained VGG19...")

    base_model = VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
        pooling="avg",
    )

    # Freeze VGG19 during the first training stage.
    base_model.trainable = False

    data_augmentation = create_data_augmentation()

    inputs = Input(
        shape=INPUT_SHAPE,
        name="endoscopy_image",
    )

    # Apply augmentation only during training.
    x = data_augmentation(inputs)

    # VGG19-specific preprocessing.
    x = preprocess_input(x)

    # Extract visual features using VGG19.
    x = base_model(
        x,
        training=False,
    )

    # Custom classifier for our GI dataset.
    x = Dense(
        256,
        activation="relu",
        name="classification_dense",
    )(x)

    x = Dropout(
        0.35,
        name="classification_dropout",
    )(x)

    outputs = Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="prediction",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="VGG19_Ulcer_Detector",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=("categorical_crossentropy"),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    return model
# ============================================================
# MODEL INFORMATION
# ============================================================


def save_model_summary(
    model: Model,
) -> None:
    """
    Save the neural-network architecture to a text file.
    """

    with MODEL_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as summary_file:

        model.summary(print_fn=lambda line: summary_file.write(line + "\n"))


# ============================================================
# SAVE CLASS INFORMATION
# ============================================================


def save_class_names() -> None:
    """
    Save class order for the Django prediction system.
    """

    class_information = {
        "class_names": CLASS_NAMES,
        "image_size": [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        ],
        "model": "VGG19",
        "number_of_classes": (len(CLASS_NAMES)),
    }

    with LABELS_PATH.open(
        "w",
        encoding="utf-8",
    ) as labels_file:

        json.dump(
            class_information,
            labels_file,
            indent=4,
        )


# ============================================================
# CALLBACKS
# ============================================================


def create_callbacks():
    """
    Create callbacks used during model training.
    """

    model_checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_learning_rate = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    )

    csv_logger = CSVLogger(TRAINING_LOG_PATH)

    return [
        model_checkpoint,
        early_stopping,
        reduce_learning_rate,
        csv_logger,
    ]


# ============================================================
# TRAINING GRAPH
# ============================================================


def save_training_graph(
    history,
) -> None:
    """
    Save training and validation accuracy/loss graphs.
    """

    training_accuracy = history.history["accuracy"]

    validation_accuracy = history.history["val_accuracy"]

    training_loss = history.history["loss"]

    validation_loss = history.history["val_loss"]

    epochs_range = range(
        1,
        len(training_accuracy) + 1,
    )

    plt.figure(figsize=(12, 5))

    # Accuracy graph
    plt.subplot(
        1,
        2,
        1,
    )

    plt.plot(
        epochs_range,
        training_accuracy,
        label="Training Accuracy",
        marker="o",
    )

    plt.plot(
        epochs_range,
        validation_accuracy,
        label="Validation Accuracy",
        marker="o",
    )

    plt.title("VGG19 Training Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(alpha=0.2)

    # Loss graph
    plt.subplot(
        1,
        2,
        2,
    )

    plt.plot(
        epochs_range,
        training_loss,
        label="Training Loss",
        marker="o",
    )

    plt.plot(
        epochs_range,
        validation_loss,
        label="Validation Loss",
        marker="o",
    )

    plt.title("VGG19 Training Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        HISTORY_IMAGE_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# TRAIN MODEL
# ============================================================


def train_model(
    model,
    train_dataset,
    validation_dataset,
):
    """
    Train the classifier attached to frozen VGG19.
    """

    callbacks = create_callbacks()

    print("\nStarting VGG19 training...")

    print(f"\nImage size: " f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}")

    print(f"Batch size: {BATCH_SIZE}")

    print(f"Maximum epochs: {EPOCHS}")

    print(f"Learning rate: " f"{LEARNING_RATE}")

    history = model.fit(
        train_dataset,
        validation_data=(validation_dataset),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    return history


# ============================================================
# VALIDATE SAVED MODEL
# ============================================================


def validate_saved_model(
    validation_dataset,
) -> None:
    """
    Load the best saved model and calculate validation accuracy.
    """

    print("\nLoading best saved model...")

    best_model = tf.keras.models.load_model(MODEL_PATH)

    print("\nEvaluating best model " "on validation dataset...")

    validation_loss, validation_accuracy = best_model.evaluate(
        validation_dataset,
        verbose=1,
    )

    print("\n" "==============================================")

    print("BEST MODEL VALIDATION RESULT")

    print("==============================================")

    print(f"Validation Loss: " f"{validation_loss:.4f}")

    print(f"Validation Accuracy: " f"{validation_accuracy * 100:.2f}%")

    print("==============================================")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """
    Main VGG19 training workflow.
    """

    print("\n" "==============================================")

    print("ULCER DETECTION - VGG19 MODEL TRAINING")

    print("==============================================")

    print(f"\nTensorFlow version: " f"{tf.__version__}")

    print("\nAvailable devices:")

    print(
        "CPU:",
        tf.config.list_physical_devices("CPU"),
    )

    print(
        "GPU:",
        tf.config.list_physical_devices("GPU"),
    )

    # Make training reproducible.
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    check_required_directories()

    (
        train_dataset,
        validation_dataset,
    ) = load_datasets()

    (
        train_dataset,
        validation_dataset,
    ) = optimize_datasets(
        train_dataset,
        validation_dataset,
    )

    model = build_model()

    print("\nModel architecture:")

    model.summary()

    save_model_summary(model)

    save_class_names()

    history = train_model(
        model,
        train_dataset,
        validation_dataset,
    )

    save_training_graph(history)

    validate_saved_model(validation_dataset)

    print("\n" "==============================================")

    print("TRAINING COMPLETED")

    print("==============================================")

    print(f"\nBest model:\n{MODEL_PATH}")

    print(f"\nClass mapping:\n{LABELS_PATH}")

    print(f"\nTraining graph:\n" f"{HISTORY_IMAGE_PATH}")

    print(f"\nTraining log:\n" f"{TRAINING_LOG_PATH}")

    print("\nNext step: evaluate the model " "using the untouched test dataset.")


if __name__ == "__main__":
    main()
