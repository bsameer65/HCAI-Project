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
    Create a class-distribution bar chart.

    Each class is represented using a different color
    and its observation count is shown above the bar.
    """

    class_counts = (
        df[target_column]
        .value_counts()
        .sort_index()
    )

    plt.figure(
        figsize=(7, 5)
    )

    # Generate one different color per class
    color_map = plt.get_cmap("tab10")

    colors = [
        color_map(i)
        for i in range(
            len(class_counts)
        )
    ]

    bars = plt.bar(
        class_counts.index.astype(str),
        class_counts.values,
        color=colors,
        alpha=0.85,
    )

    plt.xlabel(
        target_column
    )

    plt.ylabel(
        "Number of examples"
    )

    plt.title(
        f"Class Distribution — {target_column}"
    )

    plt.grid(
        axis="y",
        alpha=0.2,
    )

    # Display number above each bar
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,

            height,

            f"{int(height)}",

            ha="center",
            va="bottom",
            fontweight="bold",
        )

    return _save_figure(
        media_root,
        media_url,
        "classification_distribution",
    )


def create_score_comparison_chart(
    train_score,
    test_score,
    metric_name,
    media_root,
    media_url,
):
    """
    Create a bar chart comparing training and testing
    performance for the selected evaluation metric.
    """

    plt.figure(figsize=(6, 4))

    labels = [
        "Training",
        "Testing",
    ]

    values = [
        train_score * 100,
        test_score * 100,
    ]

    colors = [
        "#1a4f8a",
        "#2e7d32",
    ]

    bars = plt.bar(
        labels,
        values,
        color=colors,
        width=0.65,
        alpha=0.9,
    )

    plt.ylabel(
        f"{metric_name} (%)"
    )

    plt.title(
        f"Training vs Testing {metric_name}"
    )

    # Keep some space above bars for labels
    upper_limit = min(
        110,
        max(values) + 12
    )

    plt.ylim(
        0,
        upper_limit
    )

    plt.grid(
        axis="y",
        alpha=0.2,
    )

    # Add values above bars
    for bar, value in zip(
        bars,
        values,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    return _save_figure(
        media_root,
        media_url,
        "classification_score",
    )