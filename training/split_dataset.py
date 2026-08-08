from pathlib import Path
import random
import shutil

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DATASET = PROJECT_ROOT / "dataset"

OUTPUT_DATASET = PROJECT_ROOT / "dataset_split"


# ============================================================
# DATASET CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "normal",
    "esophagitis",
    "ulcerative-colitis",
    "polyps",
]


TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


RANDOM_SEED = 42


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def get_images(folder: Path) -> list[Path]:
    """
    Return all supported images from a directory.
    """

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found:\n{folder}")

    images = [
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(images)


def prepare_output_directories() -> None:
    """
    Create clean train, validation and test directories.
    """

    if OUTPUT_DATASET.exists():
        shutil.rmtree(OUTPUT_DATASET)

    split_names = [
        "train",
        "validation",
        "test",
    ]

    for split_name in split_names:

        for class_name in CLASS_NAMES:

            directory = OUTPUT_DATASET / split_name / class_name

            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


# ============================================================
# DATASET SPLITTING
# ============================================================


def split_class_images(
    class_name: str,
) -> None:
    """
    Split one disease class into:
    70% training,
    15% validation,
    15% testing.
    """

    source_directory = SOURCE_DATASET / class_name

    images = get_images(source_directory)

    total_images = len(images)

    if total_images == 0:
        raise ValueError(f"No images found for class: {class_name}")

    random.shuffle(images)

    train_count = int(total_images * TRAIN_RATIO)

    validation_count = int(total_images * VALIDATION_RATIO)

    test_count = total_images - train_count - validation_count

    train_images = images[:train_count]

    validation_images = images[train_count : train_count + validation_count]

    test_images = images[train_count + validation_count :]

    copy_images(
        train_images,
        OUTPUT_DATASET / "train" / class_name,
    )

    copy_images(
        validation_images,
        OUTPUT_DATASET / "validation" / class_name,
    )

    copy_images(
        test_images,
        OUTPUT_DATASET / "test" / class_name,
    )

    print(f"\n{class_name}")

    print(f"  Total:      {total_images}")

    print(f"  Train:      {len(train_images)}")

    print(f"  Validation: {len(validation_images)}")

    print(f"  Test:       {len(test_images)}")


def copy_images(
    images: list[Path],
    destination: Path,
) -> None:
    """
    Copy images to a split directory.
    """

    for image_path in images:

        destination_file = destination / image_path.name

        shutil.copy2(
            image_path,
            destination_file,
        )


# ============================================================
# VALIDATION
# ============================================================


def validate_split() -> None:
    """
    Validate train, validation and test image counts.
    """

    print("\n" "==============================================")

    print("DATASET SPLIT SUMMARY")

    print("==============================================")

    split_names = [
        "train",
        "validation",
        "test",
    ]

    grand_total = 0

    for split_name in split_names:

        print(f"\n{split_name.upper()}")

        split_total = 0

        for class_name in CLASS_NAMES:

            class_directory = OUTPUT_DATASET / split_name / class_name

            count = len(get_images(class_directory))

            split_total += count

            print(f"  {class_name:<20}" f"{count:>4}")

        grand_total += split_total

        print(f"  {'TOTAL':<20}" f"{split_total:>4}")

    print("\n" "----------------------------------------------")

    print(f"GRAND TOTAL: {grand_total}")

    print("==============================================")

    if grand_total == 4000:

        print("\nDataset split completed successfully.")

    else:

        print("\nWARNING: Expected 4000 images.")


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print("\n" "ULCER DETECTION - DATASET SPLITTING")

    print("==============================================")

    print(f"\nSource dataset:\n{SOURCE_DATASET}")

    print(f"\nOutput dataset:\n{OUTPUT_DATASET}")

    random.seed(RANDOM_SEED)

    print("\nCreating train / validation / test folders...")

    prepare_output_directories()

    print("\nSplitting dataset...")

    for class_name in CLASS_NAMES:

        split_class_images(class_name)

    validate_split()


if __name__ == "__main__":
    main()
