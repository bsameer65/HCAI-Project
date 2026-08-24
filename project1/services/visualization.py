import os
import uuid

import matplotlib.pyplot as plt


def _save_figure(media_root, media_url, prefix):
    """
    Save the currently active matplotlib figure using
    a unique filename and return its media URL.
    """

    filename = f"{prefix}_{uuid.uuid4().hex}.png"

    file_path = os.path.join(
        media_root,
        filename,
    )

    plt.tight_layout()

    plt.savefig(
        file_path,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close()

    return f"{media_url}{filename}"


def create_classification_scatter(
    df,
    x_column,
    y_column,
    target_column,
    media_root,
    media_url,
):
    """
    Create a scatter plot of two selected features.

    Each target class is displayed separately so matplotlib
    automatically assigns a different color to each class.
    """

    plt.figure(
        figsize=(7, 5)
    )

    classes = df[target_column].dropna().unique()

    for class_value in classes:

        class_data = df[
            df[target_column] == class_value
        ]

        plt.scatter(
            class_data[x_column],
            class_data[y_column],
            label=str(class_value),
            alpha=0.75,
        )

    plt.xlabel(x_column)
    plt.ylabel(y_column)

    plt.title(
        f"{x_column} vs {y_column}"
    )

    plt.legend(
        title=target_column
    )

    plt.grid(
        alpha=0.2
    )

    return _save_figure(
        media_root,
        media_url,
        "classification_scatter",
    )


def create_class_distribution(
    df,
    target_column,
    media_root,
    media_url,
):
    """
    Create a bar chart showing how many observations
    belong to each target class.
    """

    class_counts = (
        df[target_column]
        .value_counts()
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        class_counts.index.astype(str),
        class_counts.values,
    )

    plt.xlabel("Class")
    plt.ylabel("Number of examples")

    plt.title(
        f"Class Distribution — {target_column}"
    )

    plt.grid(
        axis="y",
        alpha=0.2,
    )

    return _save_figure(
        media_root,
        media_url,
        "classification_distribution",
    )


def create_accuracy_comparison_chart(
    train_score,
    test_score,
    media_root,
    media_url,
):
    """
    Compare training and testing scores visually.
    """

    plt.figure(
        figsize=(6, 4)
    )

    labels = [
        "Training",
        "Testing",
    ]

    values = [
        train_score * 100,
        test_score * 100,
    ]

    plt.bar(
        labels,
        values,
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.ylim(
        0,
        100,
    )

    plt.title(
        "Training vs Testing Accuracy"
    )

    plt.grid(
        axis="y",
        alpha=0.2,
    )

    return _save_figure(
        media_root,
        media_url,
        "classification_accuracy",
    )