import os
import uuid

import matplotlib.pyplot as plt


def create_classification_scatter(
    df,
    x_column,
    y_column,
    target_column,
    media_root,
    media_url,
):
    plt.figure(figsize=(7, 5))

    classes = df[target_column].unique()

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

    plt.tight_layout()

    filename = (
        f"classification_scatter_"
        f"{uuid.uuid4().hex}.png"
    )

    path = os.path.join(
        media_root,
        filename
    )

    plt.savefig(
        path,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close()

    return f"{media_url}{filename}"


def create_class_distribution(
    df,
    target_column,
    media_root,
    media_url,
):
    counts = (
        df[target_column]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        counts.index.astype(str),
        counts.values,
    )

    plt.xlabel("Class")
    plt.ylabel("Number of examples")

    plt.title(
        f"Class Distribution — {target_column}"
    )

    plt.tight_layout()

    filename = (
        f"class_distribution_"
        f"{uuid.uuid4().hex}.png"
    )

    path = os.path.join(
        media_root,
        filename
    )

    plt.savefig(
        path,
        dpi=120,
        bbox_inches="tight",
    )

    plt.close()

    return f"{media_url}{filename}"