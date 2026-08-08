from pathlib import Path
import json
import os

# Reduce TensorFlow INFO messages.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = PROJECT_ROOT / "dataset_split"

TRAIN_DIRECTORY = DATASET_ROOT / "train"

VALIDATION_DIRECTORY = DATASET_ROOT / "validation"

MODEL_DIRECTORY = PROJECT_ROOT / "model"


BASE_MODEL_PATH = MODEL_DIRECTORY / "vgg19_ulcer_disease.keras"


FINE_TUNED_MODEL_PATH = MODEL_DIRECTORY / "vgg19_ulcer_disease_finetuned.keras"


CLASS_NAMES_PATH = MODEL_DIRECTORY / "class_names.json"


FINE_TUNE_LOG_PATH = MODEL_DIRECTORY / "fine_tuning_log.csv"


FINE_TUNE_GRAPH_PATH = MODEL_DIRECTORY / "fine_tuning_history.png"


FINE_TUNE_SUMMARY_PATH = MODEL_DIRECTORY / "fine_tuned_model_summary.txt"


# ============================================================
# TRAINING SETTINGS
# ============================================================

BATCH_SIZE = 16

FINE_TUNE_EPOCHS = 6

FINE_TUNE_LEARNING_RATE = 0.00001

RANDOM_SEED = 42


# ============================================================
# CHECK FILES
# ============================================================


def check_required_files():
    """
    Check that the trained model and datasets exist.
    """

    required_paths = [
        BASE_MODEL_PATH,
        CLASS_NAMES_PATH,
        TRAIN_DIRECTORY,
        VALIDATION_DIRECTORY,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(f"\nRequired path not found:\n{path}")


# ============================================================
# LOAD CONFIGURATION
# ============================================================


def load_configuration():
    """
    Read class order and image dimensions.
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
# LOAD DATASETS
# ============================================================


def load_datasets(
    class_names,
    image_size,
):
    """
    Load train and validation datasets.
    """

    print("\nLoading training dataset...")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIRECTORY,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=image_size,
        shuffle=True,
        seed=RANDOM_SEED,
    )

    print("\nLoading validation dataset...")

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        VALIDATION_DIRECTORY,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=image_size,
        shuffle=False,
    )

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)

    return (
        train_dataset,
        validation_dataset,
    )


# ============================================================
# LOAD BASELINE MODEL
# ============================================================


def load_baseline_model():
    """
    Load the model trained in Stage 4.
    """

    print("\nLoading baseline VGG19 model...")

    model = tf.keras.models.load_model(BASE_MODEL_PATH)

    print("Baseline model loaded successfully.")

    return model


# ============================================================
# CONFIGURE FINE-TUNING
# ============================================================


def configure_fine_tuning(
    model,
):
    """
    Unfreeze only the upper layers of VGG19.

    The lower VGG19 blocks remain frozen because they
    contain useful general-purpose visual features.

    Only the final convolutional layers are adapted to
    the gastrointestinal endoscopy dataset.
    """

    try:

        vgg19_model = model.get_layer("vgg19")

    except ValueError as error:

        raise ValueError(
            "\nCould not find the VGG19 base model " "inside the saved model."
        ) from error

    # Allow the VGG19 model to contain trainable layers.
    vgg19_model.trainable = True

    # Freeze everything first.
    for layer in vgg19_model.layers:

        layer.trainable = False

    # Fine-tune only the last two convolutional layers.
    layers_to_unfreeze = {
        "block5_conv3",
        "block5_conv4",
    }

    for layer in vgg19_model.layers:

        if layer.name in layers_to_unfreeze:

            layer.trainable = True

    print("\nFine-tuning configuration:")

    print("\nVGG19 layers being trained:")

    for layer in vgg19_model.layers:

        if layer.trainable:

            print(f"  {layer.name}")

    trainable_parameters = sum(
        tf.keras.backend.count_params(weight) for weight in model.trainable_weights
    )

    non_trainable_parameters = sum(
        tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights
    )

    print(f"\nTrainable parameters: " f"{trainable_parameters:,}")

    print(f"Non-trainable parameters: " f"{non_trainable_parameters:,}")

    # IMPORTANT:
    # TensorFlow requires recompilation after modifying
    # layer trainability.

    model.compile(
        optimizer=(tf.keras.optimizers.Adam(learning_rate=(FINE_TUNE_LEARNING_RATE))),
        loss=("categorical_crossentropy"),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    return model


# ============================================================
# CALLBACKS
# ============================================================


def create_callbacks():
    """
    Callbacks for safe fine-tuning.
    """

    checkpoint = ModelCheckpoint(
        filepath=(FINE_TUNED_MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_learning_rate = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=1,
    )

    logger = CSVLogger(FINE_TUNE_LOG_PATH)

    return [
        checkpoint,
        early_stopping,
        reduce_learning_rate,
        logger,
    ]


# ============================================================
# SAVE MODEL SUMMARY
# ============================================================


def save_model_summary(
    model,
):
    """
    Save fine-tuned architecture details.
    """

    with FINE_TUNE_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        model.summary(print_fn=lambda line: file.write(line + "\n"))


# ============================================================
# TRAINING
# ============================================================


def fine_tune(
    model,
    train_dataset,
    validation_dataset,
):
    """
    Perform the second training stage.
    """

    print("\n" "==============================================")

    print("STARTING VGG19 FINE-TUNING")

    print("==============================================")

    print(f"\nMaximum epochs: " f"{FINE_TUNE_EPOCHS}")

    print(f"Learning rate: " f"{FINE_TUNE_LEARNING_RATE}")

    history = model.fit(
        train_dataset,
        validation_data=(validation_dataset),
        epochs=(FINE_TUNE_EPOCHS),
        callbacks=(create_callbacks()),
        verbose=1,
    )

    return history


# ============================================================
# SAVE FINE-TUNING GRAPH
# ============================================================


def save_training_graph(
    history,
):
    """
    Save accuracy and loss during fine-tuning.
    """

    training_accuracy = history.history["accuracy"]

    validation_accuracy = history.history["val_accuracy"]

    training_loss = history.history["loss"]

    validation_loss = history.history["val_loss"]

    epoch_numbers = range(
        1,
        len(training_accuracy) + 1,
    )

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(
        1,
        2,
        1,
    )

    plt.plot(
        epoch_numbers,
        training_accuracy,
        marker="o",
        label="Training Accuracy",
    )

    plt.plot(
        epoch_numbers,
        validation_accuracy,
        marker="o",
        label="Validation Accuracy",
    )

    plt.title("VGG19 Fine-Tuning Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(alpha=0.2)

    # Loss
    plt.subplot(
        1,
        2,
        2,
    )

    plt.plot(
        epoch_numbers,
        training_loss,
        marker="o",
        label="Training Loss",
    )

    plt.plot(
        epoch_numbers,
        validation_loss,
        marker="o",
        label="Validation Loss",
    )

    plt.title("VGG19 Fine-Tuning Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        FINE_TUNE_GRAPH_PATH,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# VALIDATE FINE-TUNED MODEL
# ============================================================


def evaluate_validation_dataset(
    validation_dataset,
):
    """
    Load the best fine-tuned model and evaluate it.
    """

    print("\nLoading best fine-tuned model...")

    fine_tuned_model = tf.keras.models.load_model(FINE_TUNED_MODEL_PATH)

    print("\nEvaluating fine-tuned model " "on validation data...")

    results = fine_tuned_model.evaluate(
        validation_dataset,
        verbose=1,
        return_dict=True,
    )

    print("\n" "==============================================")

    print("FINE-TUNED VALIDATION RESULT")

    print("==============================================")

    print(f"Validation Loss: " f"{results['loss']:.4f}")

    print(f"Validation Accuracy: " f"{results['accuracy'] * 100:.2f}%")

    print("==============================================")


# ============================================================
# MAIN
# ============================================================


def main():

    print("\n" "==============================================")

    print("ULCER DETECTION - VGG19 FINE-TUNING")

    print("==============================================")

    print(f"\nTensorFlow version: " f"{tf.__version__}")

    print(
        "\nCPU:",
        tf.config.list_physical_devices("CPU"),
    )

    print(
        "GPU:",
        tf.config.list_physical_devices("GPU"),
    )

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    check_required_files()

    (
        class_names,
        image_size,
    ) = load_configuration()

    print("\nClass order:")

    for index, class_name in enumerate(class_names):

        print(f"  {index} -> {class_name}")

    print(f"\nImage size: " f"{image_size[0]} x " f"{image_size[1]}")

    (
        train_dataset,
        validation_dataset,
    ) = load_datasets(
        class_names,
        image_size,
    )

    model = load_baseline_model()

    model = configure_fine_tuning(model)

    print("\nFine-tuned model architecture:")

    model.summary()

    save_model_summary(model)

    history = fine_tune(
        model,
        train_dataset,
        validation_dataset,
    )

    save_training_graph(history)

    evaluate_validation_dataset(validation_dataset)

    print("\n" "==============================================")

    print("FINE-TUNING COMPLETED")

    print("==============================================")

    print("\nBaseline model:")

    print(BASE_MODEL_PATH)

    print("\nFine-tuned model:")

    print(FINE_TUNED_MODEL_PATH)

    print("\nFine-tuning log:")

    print(FINE_TUNE_LOG_PATH)

    print("\nFine-tuning graph:")

    print(FINE_TUNE_GRAPH_PATH)


if __name__ == "__main__":
    main()
