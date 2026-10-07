# Store Sales LSTM

I'm working on a sales forecasting project using the Kaggle **Store Sales - Time Series Forecasting** data. The goal is to predict daily sales for each store and product family in Ecuador, using a global LSTM model.

This is still a work in progress. I've started on loading the data and creating features, but the model, dataset, training, prediction and evaluation code isn't in place yet. So there isn't a full training or prediction workflow to run at the moment.

## Data

Download the competition data and put the CSVs here:

```text
data/raw/store_sales/
```

The data loader currently expects `train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv` and `transactions.csv`. `sample_submission.csv` is also in the data folder, but it isn't used by the loader yet.

## Setup

Install the packages from `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

## What I've Started

- `src/store_sales/config.py` has the current project and training defaults.
- `src/store_sales/data.py` loads the CSV files and has a few basic data checks.
- `src/store_sales/features.py` has early work on calendar and holiday features, panel data, temporal splits, and normalization. It still needs work before it can be used as a complete pipeline.
- `notebooks/` has the EDA and LSTM experiment notebooks.

The defaults in `Config` include a 90-day context length, batch size of 512, hidden size of 128, 2 LSTM layers, and up to 30 training epochs. `configs/baseline.yaml` is empty for now.

## Project Folders

```text
configs/                 Configuration files
data/raw/store_sales/    Competition data
data/interim/            Intermediate data
data/processed/          Processed data
artifacts/               Models, predictions, and preprocessing outputs
notebooks/               EDA and experiments
src/store_sales/         Project code
tests/                   Tests (not written yet)
```
