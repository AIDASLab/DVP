import pandas as pd
import os
from pathlib import Path

def load_local_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Load a local dataset and return a DataFrame with a 'text' column.

    Args:
        dataset_name (str): Name of the dataset to load.

    Returns:
        pd.DataFrame: DataFrame containing at least the 'text' column.

    Raises:
        AssertionError: If the 'text' column is missing.
        ValueError: If dataset_name is not recognized.
    """
    base_dir = os.environ.get(
        "LLM_VALUE_DATA_ROOT", str(Path(__file__).resolve().parents[1] / "data")
    )

    if dataset_name == "culturax_ko":
        path = f"{base_dir}/culturax/ko_100K.csv"
        assert os.path.exists(path), f"File not found: {path}"
        data = pd.read_csv(path)
        assert "text" in data.columns, "'text' column not found in culturax_ko dataset"
        return data

    elif dataset_name.startswith("culturax_"):
        lang = dataset_name.split("_")[1]
        path = f"{base_dir}/culturax/{lang}_100K.csv"
        assert os.path.exists(path), f"File not found: {path}"
        data = pd.read_csv(path)
        assert "text" in data.columns, f"'text' column not found in {dataset_name} dataset"
        return data

    elif dataset_name == "prism_utterances":
        path = f"{base_dir}/prism/utterances.csv"
        assert os.path.exists(path), f"File not found: {path}"
        data = pd.read_csv(path)
        assert "text" in data.columns, "'text' column not found in prism_utterances dataset"
        return data

    elif dataset_name == "prism_conversations":
        path = f"{base_dir}/prism/conversations.csv"
        assert os.path.exists(path), f"File not found: {path}"
        data = pd.read_csv(path)
        assert "text" in data.columns, "'text' column not found in prism_conversations dataset"
        return data

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
