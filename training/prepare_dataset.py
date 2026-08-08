from pathlib import Path
import random
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

# Original downloaded Kvasir v2 dataset.
SOURCE_DATASET = Path(
    r"C:\Users\Dharani G\Downloads\kvasir-dataset-v2\kvasir-dataset-v2"
)

# Balanced dataset that will be created inside our Django project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DATASET = PROJECT_ROOT / "dataset"


# We use a fixed random seed so the same images are selected
# whenever we prepare the dataset again.
RANDOM_SEED = 42


# Supported image file extensions.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# DATASET SETTINGS
# ============================================================

CLASS_TARGET_COUNT = 1000


NORMAL_CLASS_SOURCES = {
    "normal-cecum": 333,
    "normal-pylorus": 333,
    "normal-z-line": 334,
}


DIRECT_CLASSES = {
    "esophagitis": 1000,
    "ulcerative-colitis": 1000,
    "polyps": 1000,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def get_image_files(folder: Path) -> list[Path]:
    """
    Return all supported image files from a folder.
    """

    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist:\n{folder}")

    image_files = [
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(image_files)


def prepare_output_directory() -> None:
    """
    Create the four output class directories.

    Existing class folders are removed first so the generated
    dataset is always clean and reproducible.
    """

    classes = [
        "normal",
        "esophagitis",
        "ulcerative-colitis",
        "polyps",
    ]

    OUTPUT_DATASET.mkdir(
        parents=True,
        exist_ok=True,
    )

    for class_name in classes:

        class_directory = OUTPUT_DATASET / class_name

        if class_directory.exists():
            shutil.rmtree(class_directory)

        class_directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def select_random_images(
    folder: Path,
    amount: int,
) -> list[Path]:
    """
    Select a fixed number of random images from a folder.
    """

    images = get_image_files(folder)

    if len(images) < amount:

        raise ValueError(
            f"Not enough images in {folder.name}.\n"
            f"Required: {amount}\n"
            f"Available: {len(images)}"
        )

    return random.sample(
        images,
        amount,
    )


def copy_images(
    images: list[Path],
    destination: Path,
    filename_prefix: str,
) -> None:
    """
    Copy selected images into the destination directory.

    Files are renamed to prevent duplicate filenames.
    """

    for index, source_file in enumerate(
        images,
        start=1,
    ):

        extension = source_file.suffix.lower()

        new_filename = f"{filename_prefix}_{index:04d}{extension}"

        destination_file = destination / new_filename

        shutil.copy2(
            source_file,
            destination_file,
        )


# ============================================================
# NORMAL CLASS
# ============================================================


def create_normal_class() -> None:
    """
    Build one balanced 'normal' category using images from:

    - normal-cecum
    - normal-pylorus
    - normal-z-line
    """

    print("\nCreating NORMAL class...")

    destination = OUTPUT_DATASET / "normal"

    total_copied = 0

    for source_class, amount in NORMAL_CLASS_SOURCES.items():

        source_directory = SOURCE_DATASET / source_class

        print(f"Selecting {amount} images " f"from {source_class}...")

        selected_images = select_random_images(
            source_directory,
            amount,
        )

        copy_images(
            selected_images,
            destination,
            source_class,
        )

        total_copied += amount

    print(f"Normal class completed: " f"{total_copied} images")


# ============================================================
# DISEASE CLASSES
# ============================================================


def create_direct_classes() -> None:
    """
    Create esophagitis, ulcerative-colitis and polyps classes.
    """

    for class_name, amount in DIRECT_CLASSES.items():

        print(f"\nCreating {class_name.upper()} class...")

        source_directory = SOURCE_DATASET / class_name

        destination = OUTPUT_DATASET / class_name

        selected_images = select_random_images(
            source_directory,
            amount,
        )

        copy_images(
            selected_images,
            destination,
            class_name,
        )

        print(f"{class_name} completed: " f"{amount} images")


# ============================================================
# VALIDATION
# ============================================================


def validate_dataset() -> None:
    """
    Print and validate the number of generated images.
    """

    print("\n" "============================================")

    print("FINAL DATASET SUMMARY")

    print("============================================")

    expected_classes = {
        "normal": CLASS_TARGET_COUNT,
        "esophagitis": CLASS_TARGET_COUNT,
        "ulcerative-colitis": CLASS_TARGET_COUNT,
        "polyps": CLASS_TARGET_COUNT,
    }

    total_images = 0

    dataset_valid = True

    for class_name, expected_count in expected_classes.items():

        class_directory = OUTPUT_DATASET / class_name

        actual_count = len(get_image_files(class_directory))

        total_images += actual_count

        status = "OK" if actual_count == expected_count else "ERROR"

        if actual_count != expected_count:
            dataset_valid = False

        print(f"{class_name:<20}" f"{actual_count:>5} images   " f"[{status}]")

    print("--------------------------------------------")

    print(f"{'TOTAL':<20}" f"{total_images:>5} images")

    print("============================================")

    if dataset_valid:

        print("\nDataset preparation completed successfully.")

    else:

        print("\nDataset validation failed.")


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print("\nULCER DETECTION - DATASET PREPARATION")

    print("============================================")

    print(f"Source dataset:\n{SOURCE_DATASET}")

    print(f"\nOutput dataset:\n{OUTPUT_DATASET}")

    print("\nPreparing balanced 4-class dataset...")

    random.seed(RANDOM_SEED)

    prepare_output_directory()

    create_normal_class()

    create_direct_classes()

    validate_dataset()


if __name__ == "__main__":
    main()
