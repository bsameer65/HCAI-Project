from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame


def _to_dataframe(dataset_split) -> pd.DataFrame:
    dataframe = dataset_split.to_pandas()[["text", "label"]].copy()

    dataframe["text"] = (
        dataframe["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe["label"] = dataframe["label"].astype(int)

    if dataframe.empty:
        raise ValueError("The dataset split is empty.")

    if dataframe["text"].eq("").any():
        raise ValueError("The dataset contains empty text values.")

    return dataframe.reset_index(drop=True)


def load_ag_news() -> DatasetBundle:
    dataset = load_dataset("fancyzhx/ag_news")

    return DatasetBundle(
        train=_to_dataframe(dataset["train"]),
        test=_to_dataframe(dataset["test"]),
    )