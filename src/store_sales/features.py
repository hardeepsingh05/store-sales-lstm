from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

# ======================================
# Feature definitions
# ======================================

SERIES_COL = [
    "store_nbr",
    "family",
]

CAT_COLS = [
    "store_nbr",
    "family",
    "city",
    "state",
    "store_type",
    "store_cluster"
]

STATIC_CAT_FEATURES = [
    f"{col}_idx" for col in CAT_COLS
]

STANDARDIZE_COLS = [
    "log_onpromotion",
    "oil_price",
    "log_transactions",
    "time_idx"
]

HIST_FEATURES = [
    # Historical target
    "target_z",

    # Historical numerical/business information
    "log_onpromotion_z",
    "oil_price_z",
    "log_transactions_z",
    "time_idx_z",

    # Calender
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_payday",

    # Holidays/events
    "special_national",
    "special_regional",
    "special_local",
    "event_flag"
]

FUTURE_FEATURES = [
    # Known future covariates
    "log_onpromotion_z",
    "oil_price_z",
    "time_idx_z",

    # Calender
    "dow_sin",
    "dow_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_payday",

    # Holidays/events
    "special_national",
    "special_regional",
    "special_local",
    "event_flag"
]

# ==================================
# DataClasses
# ==================================

@dataclass(frozen=True)
class TemporalSplit:
    horizon: int
    fit_end: pd.Timestamp
    val_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp

@dataclass
class PreprocessorState:
    numeric_stats: dict[str, dict[str, float]]
    vocabs: dict[str, dit[str, int]]
    target_stats: pd.DataFrame
    global_target_loc: float
    global_taget_scale: float
    fit_cutoff: pd.Timestamp

# ======================================
# Temporal split
# =====================================

def get_temporal_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> TemporalSplit:
    """
    Creates a validation period having the same number of days as the Kaggle test period.

    Args:
        train: pd.DataFrame = training data
        test: pd.DataFrame = testing data

    Returns:
        TemporalSplit: DataClass
    """

    horizon = int(test['date'].nunique())
    if horizon <= 0:
        raise ValueError("Test horizon must be greater than zero.")
    
    train_end = pd.Timestamp(
        train['date'].max()
    )

    test_start = pd.Timestamp(
        test['date'].min()
    )

    val_start = (
        train_end - pd.Timedelta(days=horizon - 1)
    )

    fit_end = (
        val_start - pd.Timedelta(days=1)
    )

    return TemporalSplit(
        horizon=horizon,
        fit_end=fit_end,
        val_start=val_start,
        train_end=train_end,
        test_start=test_start
    )

# =======================================
# Holiday helper 
# =======================================

def _binary_flag_table(
    df: pd.DataFrame,
    keys: list[str],
    flag_name: str
    ) -> pd.DataFrame:
    """
    Converts potentially multiple holiday rows into one binary indicator per key combination.

    Args:
        df: pd.DataFrame = Input pandas dataframe
        keys: list[str] = List of categories/keys whose rows count needs to be checked in dataframe
        flag_name: str = rename to newly added column
    
    Returns:
        pd.DataFrame: pandas dataframe after reseting index from df.groupby()
    """

    if df.empty:
        columns = [*keys, flag_name]
        return pd.DataFrame(columns=columns)

    result = (
        df.groupby(keys)
        .size()
        .gt(0)
        .astype("float32")
        .rename(flag_name)
        .reset_index()
    )

    return result

def prepare_holiday_features(
    holidays: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """
    Separates holidays according to their geographic scope. 
    Transferred=True means that particular original holiday date is not the effective holiday date, so it's excluded.

    Args: 
        holidays: pd.DataFrame = dataframe related to holidays.
    
    Returns:
        A dictionary containing flag pandas dataframes.
    """

    hol = holiday.copy()

    hol['transferred'] = (
        hol['transferred'].fillna(False).astype(bool)
    )

    active = hol.loc[~hol['transferred']].copy()

    # Events receive their own flag
    events = active.loc[active['type'].eq("Event")]
    event_flag = _binary_flag_table(
        events,
        keys = ['date'],
        flag_name="event_flag",
    )

    # Work Day means a day that would otherwise normally be considered a holiday/weekend
    # but is designed as work. We don't treat it as a holiday.
    special = active.loc[
        ~active["type"].isin(['Event','Work Day'])
    ].copy()

    national = special.loc[special['locale'].eq("National")]
    national_flag = _binary_flag_table(
        national,
        keys=['date'],
        flag_name="special_national"
    )

    regional = (
        special.loc[special['locale'].eq("Regional")].rename(
            columns = {
                "locale_name": "state"
            }
        )
    )

    regional_flag = _binary_flag_table(
        regional,
        keys = ["date","state"],
        flag_name="special_regional",
    )

    local = (
        special.loc[
            special["locale"].eq("Local")
        ].rename(
            columns = {
                "locale_name":"city"
            }
        )
    )

    local_flag = _binary_flag_table(
        local,
        keys = ["date","city"],
        flag_name="special_local",
    )

    return {
        "national": national_flag,
        "regional": regional_flag,
        "local": local_flag,
        "events": event_flag,
    }

# =================================================
# Calendar features
# =================================================

def add_calender_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A function/method to add new columns in dataset based on "date" column.

    Args:
        df: pd.DataFrame = Input pandas DataFrame to get new feature based on "date" column added.

    Returns:
        df: pd.DataFrame = Pandas DataFrame included of newly added features.
    """

    df = df.copy()
    date_ = df["date"].dt

    df['day_of_week'] = (date_.dayofweek.astype("int16"))
    df['day_of_year'] = (date_.dayofyear.astype("int16"))

    # Cyclical encoding

    df['dow_sin'] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    ).astype("float32")

    df['dow_cos'] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    ).astype("float32")

    df['doy_sin'] = np.sin(
        2 * np.pi * df["day_of_year"] / 365.25
    ).astype("float32")

    df['doy_cos'] = np.cos(
        2 * np.pi * df["day_of_year"] / 365.25
    ).astype("float32")

    # Other Calendar Signals
    df["is_weekened"] = (
        df['day_of_week'] >= 5
    ).astype("float32")

    df['is_payday'] = (
        (date_.day == 15) | date_.is_month_end
    ).astype("float32")

    # Continous trend
    start_date = df['date'].min()
    df['time_idx'] = (
        df['date'] - start_date
    ).dt.days.astype("float32")

    return df

# =====================================
# Base Panel Construction
# =====================================

def build_feature_panel(
    data: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
    """
    Constructs one unified Store X Family X Date Panel. No train-fitted statistics are calculated here.
    Therefor this step can safely be shared between training and inference.

    Args:
        data: dict[str, pd.DataFrame] = A dictionary containing all the dataframe.

    Returns:
        Final pandas dataframe containing all the features.
    """ 

    train =  data['train'].copy()
    test = data['test'].copy()
    stores = data['stores'].copy()
    oil = data['oil'].copy()
    holidays = data['holidays'].copy()
    transactions = data['transactions'].copy()

    # Train + Test Data
    train['is_train'] == 1
    test['is_train'] == 0
    test['sales'] = np.nan

    panel = pd.concat(
        [train, test], ignore_index = True, sort = False
    )

    original_row_count = len(panel)

    # Store metadata
    stores = stores.rename(
        columns = {
            "type": "store_type",
            "cluster": "store_cluster"
        }
    )
    # merging stores data into panel dataset 
    panel = panel.merge(
        stores,
        on="store_nbr",
        how="left",
        validate = "many_to_one",
    )

    if len(panel) != original_row_count:
        raise RuntimeError(
            "Store merge unexpectedly changed row count."
        )

    # Oil data
    all_dates = pd.DataFrame(
        {
            "dates": pd.date_range(
                panel['date'].min(),
                panel['date'].max(),
                freq="D",
            )
        }
    )

    oil_daily = all_dates.merge(
        oil,
        on="date",
        how="left",
        validate="one_to_one",
    )

    oil_daily = oil_daily.rename(
        columns={
            "dcoilwtico":"oil_price"
        }
    )

    # Causal filling: only use previously known oil value.
    oil_daily['oil_price']=(
        oil_daily['oil_price'].ffill()
    )

    panel = panel.merge(
        oil_daily,
        on="date",
        how="left",
        validate="many_to_one",
    )

    # Transactions
    panel = panel.merge(
        transactions,
        on=[
            "date",
            "store_nbr"
        ],
        how="left",
        validate="many_to_one"
    )

    # Missing transactions generally indicate no recorded transaction activity
    panel['transactions'] = (
        panel['transaction'].fillna(0).astype("float32")
    )

    # Holidays
    holiday_tables = prepare_holiday_features(
        holidays=holidays
    )

    panel = panel.merge(
        holiday_tables['national'],
        on="date",
        how="left",
        validate="many_to_one",
    )

    panel = panel.merge(
        holiday_tables['regional'],
        on=[
            "date",
            "state"
        ],
        how="left",
        validate="many_to_one"
    )

    panel = panel.merge(
        holiday_tables['local'],
        on=[
            "date",
            "state"
        ],
        how="left",
        validate="many_to_one",
    )

    panel = panel.merge(
        holiday_tables['events'],
        on="date",
        how="left",
        validate="many_to_one",
    )

    holiday_cols = [
        "special_national",
        "special_regional",
        "special_local",
        "event_flag",
    ]

    panel[holiday_cols] = (
        panel[holiday_cols].fillna(0).astype("float32")
    )

    # Calendar features
    panel = add_calender_features(
        df=panel
    )

    # Log transformations
    panel["sales_log"]=np.log1p(
        panel['sales']
    )

    panel["log_onpromotiona"] = np.log1p(
        panel["onpromotion"].clip(lower=0)
    ).astype("float32")

    panel["log_transactions"] = np.log1p(
        panel["transaction"].clip(lower=0)
    ).astype("float32")

    # Sorting
    panel = (
        panel.sort_values(
            [
                "store_nbr",
                "family",
                "date",
            ]
        ).reset_index(drop=True)
    )

    if len(panel) != original_row_count:
        raise RuntimeError(
            "Feature construction changed panel row count."
        )

    return panel


# =========================================
# Numerical preprocessing
# =========================================

def fit_numeric_stats(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    columns: list[str] = STANDARDIZE_COLS,
    ) -> dict[str, dict[str,float]]:
    """
    Calculate preprocessing statistics using only data through cutoff.

    Args:
        df: Input pandas dataframe
        cutoff: Timestamp to cut-off the data rows.
        columns: List of columns in input dataframe to used for numeric stats
    
    Returns:
        Dictionary consisting stats(median, mean, standard deviation) for respective columns.
    """

    mask = (
        (df['date'] <= cutoff) & df['sales'].notna()
    )

    fit_df = df.loc[mask]
    stats: dict[str, dict[str, float]] = {}

    for column in columns:
        values = (
            fit_df[col].astype("float64")
        )

        median = values.median()
        if pd.isna(median):
            median = 0.0
        filled = values.fillna(median)
        
        mean = filled.mean()

        std = filled.std(
            ddof=0              # Divides the sum of squared deviations by N
        )
        if (pd.isna(std) or std < 1e-8):
            std = 1.0
        
        stats[col] = {
            "median": float(median),
            "mean": float(mean),
            "std": float(std)
        }

    return stats


def apply_numeric_stats(
    df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
    """
    Applying already calculated stats over to respective columns in dataframe

    Args:
        df: Input pandas dataframe 
        stats: Dictionary consisting stats(median, mean, standard deviation) for respective columns in input dataframe.

    Returns:
        Pandas dataframe after numeric stats have been applied to input dataframe.
    """

    df = df.copy()

    for col, values in stats.items():
        x = (
            df[col].astype("float64").fillna(values['median'])
        )

        df[f"{col}_z"] = (
            (x - values['mean']) / values['std']
        ).astype("float32")

    return df


# ============================================
# Target Normalization 
# ============================================

def fit_target_stats(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, float, float]:
    """
    Calculate preprocessing statistics for target column through cutoff.

    Args:
        df: Input pandas dataframe.
        cutoff: Timestamp to cut-off the data rows.
    
    Returns:
        A tuple containing
    """

    mask = (
        (df['date'] <= cutoff) & df['sales_log'].notna()
    )

    fit_df = df.loc[
        mask, [*SERIES_COLS, "sales_log",],
    ]
    target_stats = (
        fit_df.groupby(
            SERIES_COL, as_index=False,
        ).agg(
            target_loc=(
                "sales_log",
                "mean"
            ), 
            target_scale = (
                "sales_log",
                lambda x: x.std(ddof=0),
            ),
        )
    )

    target_stats["target_scale"] = (
        target_stats["target_scale"].fillna(0.1).clip(lower=0.1)
    )

    global_target_loc = float(fit_df["sales_log"].mean())
    global_target_scale = float(fit_df["sales_log"].std(ddof=0))

    if (pd.isna(global_target_scale) or global_target_scale < 0.1):
        global_target_scale = 1.0
    
    return (
        target_stats,
        global_target_loc,
        global_target_scale,
    )

def apply_target_stats(
    df: pd.DataFrame,
    target_stats: pd.DataFrame,
    global_target_loc: float,
    global_target_scale: float,
    ) -> pd.DataFrame:

    """
    Applying already calculated stats over to target column in dataframe

    Args:
        df: Input pandas dataframe 
        target_stats: Pandas dataframe containing target column stats,
        global_target_loc: mean of normalized taget column
        global_target_scale: standard deviation of normalized target column

    Returns:
        Pandas dataframe after numeric stats have been applied to input dataframe and created some new columns.
    """
    df = df.copy()

    df = df.drop(
        columns = [
            "target_loc",
            "target_scale",
            "target_z",
        ], errors="ignore",
    )

    df = df.merge(
        target_stats,
        on=SERIES_COL,
        how ="left",
        validate="many_to_one",
    )

    df["target_loc"] = (
        df["target_loc"].fillna(global_target_loc).astype("float32")
    )
    df["target_scale"] = (
        df["target_scale"].fillna(global_target_scale).clip(lower=0.1).astype("float32")
    )
    df["target_z"] = (
        (df["sales_log"] - df["target_loc"])  / df["target_scale"]
    ).astype("float32")

    return df

# ====================================
# Categorial Vocabularies
# ====================================

def _normalize_category(
    series: pd.Series,
    ) -> pd.Series:
    """
    Normalizing a pandas series or categorical columns

    Args:
        series: pandas series to normalize

    Returns:
        Pandas series after normalizing
    """

    return (
        series.astype("string").fillna("<MISSING>").astype(str)
    )

def fit_vocab(
    series: pd.Series,
    ) -> dict[str, int]:
    """
    Creating a vocabulary mapping for a categorical pandas series.

    Args:
        series: pandas series to create vocabulary for

    Returns:
        Dictionary mapping each unique category to a unique integer, starting from 1. 0 is reserved for UNKNOWN categories.
    """

    values = (
        _normalize_category(series).drop_duplicates().sort_values().tolist()
    )

    # 0 is reserved for UNKNOWN categories.
    return {
        value: idx + 1 for idx, value in enumerate(values)
    }

def apply_vocab(
    series: pd.Series,
    vocab: dict[str,int],
) -> pd.Series:
    """
    Applying a vocabulary mapping to a categorical pandas series.

    Args:
        series: pandas series to apply vocabulary to
        vocab: Dictionary mapping each unique category to a unique integer

    Returns:
        Pandas series with categories replaced by their corresponding integer values. Unknown categories are mapped to 0.
    """
    
    normalized = _normalize_category(series)
    return normalized.map(vocab).fillna(0).astype("int64")

def fit_vocabs(
    df: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> dict[str, dict[str, int]]:
    """
    Fitting vocabularies for all categorical columns in a DataFrame up to a given cutoff date.

    Args:
        df: pandas DataFrame containing the data
        cutoff: Timestamp to filter the data up to

    Returns:
        Dictionary mapping each categorical column to its vocabulary dictionary
    """

    mask = (
        df["date"] <= cutoff
    )  

    vocabs = {}
    
    for col in CAT_COLS:
        vocabs[col] = fit_vocab(df.loc[mask, col])
    return vocabs

def apply_vocabs(
    df: pd.DataFrame,
    vocabs: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """
    Applying vocabularies to all categorical columns in a DataFrame.

    Args:
        df: pandas DataFrame containing the data
        vocabs: Dictionary mapping each categorical column to its vocabulary dictionary

    Returns:
        DataFrame with categorical columns replaced by their corresponding integer values
    """

    df = df.copy()
    for col, vocab in vocabs.items():
        df[f"{col}_idx"] = apply_vocab(df[col], vocab)
    return df


# ================================================
# Full Preprocessor
# ================================================

def fit_preprocessor(
    panel: pd.DataFrame,
    fit_cutoff: pd.Timestamp,
) -> PreprocessorState:
    """
    Fitting the full preprocessor for the given DataFrame up to a specified cutoff date.

    Args:
        panel: pandas DataFrame containing the data
        fit_cutoff: Timestamp to filter the data up to

    Returns:
        PreprocessorState containing the fitted vocabularies and any other preprocessing state
    """
    
    numeric_stats = fit_numeric_stats(panel, fit_cutoff)
    
    (target_stats, global_target_loc, global_target_scale) = fit_target_stats(panel, fit_cutoff)

    vocabs = fit_vocabs(panel, fit_cutoff)

    return PreprocessorState(
        numeric_stats=numeric_stats,
        vocabs=vocabs,
        target_stats=target_stats,
        global_target_loc=global_target_loc,
        global_target_scale=global_target_scale,
        fit_cutoff=pd.Timestamp(fit_cutoff),
    )

def transform_panel(
    panel: pd.DataFrame,
    state: PreprocessorState,
) -> pd.DataFrame:
    """
    Transforming the given panel DataFrame using the fitted preprocessor state.

    Args:
        panel: pandas DataFrame containing the data
        state: PreprocessorState containing the fitted vocabularies and preprocessing state

    Returns:
        Transformed DataFrame with numeric columns standardized and categorical columns replaced by their corresponding integer values
    """

    df = apply_numeric_stats(panel, state.numeric_stats)
    
    df = apply_target_stats(
        df, 
        target_stats=state.target_stats, 
        global_target_loc=state.global_target_loc, 
        global_target_scale=state.global_target_scale
    )
   
    df = apply_vocabs(df, state.vocabs)

    return df

def fit_transform_panel(
    panel: pd.DataFrame,
    fit_cutoff: pd.Timestamp,
) -> Tuple[pd.DataFrame, PreprocessorState]:
    """
    Fit the preprocessor on the given panel DataFrame up to a specified cutoff date and transform the panel.

    Args:
        panel: pandas DataFrame containing the data
        fit_cutoff: Timestamp to filter the data up to

    Returns:
        A tuple containing the transformed DataFrame and the fitted PreprocessorState
    """
    state = fit_preprocessor(
        panel, 
        fit_cutoff=fit_cutoff
        )
    transformed_panel = transform_panel(panel, state)

    validate_model_frame(transformed_panel)

    return transformed_panel, state

# =============================
# Validation
# =============================

def validate_model_frame(
    df: pd.DataFrame,
) -> None:
    """
    Validate the given DataFrame to ensure it contains all required columns and no missing values in critical features.

    Args:
        df: pandas DataFrame containing the data to validate

    Raises:
        ValueError: If any required columns are missing or if there are NaN values in critical features
    """
    required_columns = set(
        HIST_FEATURES + FUTURE_FEATURES + STATIC_CAT_FEATURES
        )
    missing_columns = (required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
            f"{sorted(missing_columns)}"
            )
    
    # target_z is intentionally NaN in test rows
    numeric_features = list(
        set(
            HIST_FEATURES + FUTURE_FEATURES) - {"target_z"}
        )
    
    if (df[numeric_features].isna().any().any()):
        bad = (df[numeric_features].isna().sum())
        bad = bad[bad > 0]
        raise ValueError(
            f"NaN found in model features: {bad}"
        )
    observed_mask = (
        df["sales_log"].notna()
    )
    
    if (df.loc[observed_mask, "target_z",].isna().any()):
        raise ValueError(
            "Observed target rows contain NaN target_z."
        )
    if (
        df[STATIC_CAT_FEATURES].isna().any().any()
    ):
        raise ValueError(
            "Missing values found in static categorical features."
        )

# ===============================
# Cardinalities Checks
# ==============================
def get_categorical_cardinalities(
    state: PreprocessorState,
) -> list[int]:
    """
    Get the cardinalities of the categorical features based on the preprocessor state.

    Args:
        state: PreprocessorState containing the vocabulary information for categorical features

    Returns:
        A list of integers representing the cardinalities of the categorical features
    """
    return [
        len(state.vocabs[col]) + 1 for col in CAT_COLS
    ]

# =================================
# Artifact persistence
# =================================
def save_preprocessor(
    state: PreprocessorState,
    path: Path,
) -> None:
    """
    Save the preprocessor state to a file.

    Args:
        state: PreprocessorState to be saved
        path: Path to the file where the state should be saved

    Returns:
        None
    """
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(state, f)

def load_preprocessor(
    path: Path,
) -> PreprocessorState:
    """
    Load the preprocessor state from a file.

    Args:
        path: Path to the file containing the saved preprocessor state

    Returns:
        PreprocessorState object loaded from the file
    """
    path = Path(path)

    with path.open("rb") as f:
        state = pickle.load(f)
    
    if not isinstance(state, PreprocessorState):
        raise ValueError(
            f"Loaded artifact is not a PreprocessorState: {type(state)}"
        )
    
    return state

    

   












