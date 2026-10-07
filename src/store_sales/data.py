## Loading important 
from pathlib import Path
import pandas as pd

def load_raw_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """
    Load raw dataset from the directory.

    Args: 
        data_dir: path to the raw data directory.

    Returns:
        dict[str, pd.DataFrame]: A dictionary containing all the datasets loaded under specified variables.
    """

    datasets = {
        "train": pd.read_csv(
            data_dir / "train.csv",
            parse_dates=["date"]
        ),
        "test": pd.read_csv(
            data_dir / "test.csv",
            parse_dates=["date"]
        ),
        "stores": pd.read_csv(
            data_dir / "stores.csv",
            parse_dates=["date"]
        ),
        "oil": pd.read_csv(
            data_dir / "oil.csv",
            parse_dates=["date"]
        ),
        "holidays": pd.read_csv(
            data_dir / "holidays_events.csv",
            parse_dates=["date"]
        ),
        "transactions": pd.read_csv(
            data_dir / "transactions.csv",
            parse_dates = ["date"]
        )
    }

    return datasets

def validate_raw_data(data: dict[str, pd.DataFrame]) -> None:
    """
    A function to help verify the certainity of datasets.

    Args:
        data: dict[str, pd.DataFrame] = A combined dictionary of all datasets.

    Returns: 
        None
    """
    train = data["train"]
    test = data["test"]

    assert not train.empty, "Train data should not be empty."
    assert not test.empty, "Test data should not be empty."

    assert train['sales'].notna().all(), "Train data contains missing values in the 'sales' column."
    assert (train['sales'] >= 0).all(), "Train data contains negative values in the 'sales' column."

    assert not train.duplicated(
        ["date", "store_nbr", "family"]
    ).any(), "Train data contains duplicate rows for the same date, store, and family combination."
    
    assert train["date"].max() < test["date"].min(), "Train data's max date should be earlier than test data's min date."

def audit_dateframe(name: str, df: pd.DataFrame) -> None:
    """
    A function to audit the common stats related to any dataframe
    
    Args:
        name: str = 'name' of the dataframe.
        df : pd.DataFrame = placeholder for acutal dataset in Pandas DataFrame().

    Returns:
        None
    """
    
    print(f"\n {name}")
    print("-" * 60)

    print("Shape: ", df.shape)

    print("\n Data Types: ")
    print(df.dtypes)

    print("\nMissing Data Points: ")
    print(
        df.isna()
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )

    print("\nDuplicates:", df.duplicates().sum())


