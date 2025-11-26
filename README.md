# House Price Prediction — Feature Engineering

This repository contains a reproducible pipeline for improving house price predictions using feature engineering and a baseline RandomForest model.

## Structure
- `src/` — Python source code
  - `house_price_pipeline.py` — main runnable pipeline
  - `feature_engineering.py` — preprocessing & encoding helpers
- `notebooks/` — a starter Jupyter notebook (illustrative)
- `requirements.txt` — dependencies
- `README.md` — this file

## Quick start
1. Place `train.csv`, `test.csv`, and `sample_submission.csv` into the project root or update paths in `src/house_price_pipeline.py`.
2. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt
```
3. Run the pipeline:
```bash
python src/house_price_pipeline.py
```
Outputs will be saved into `output/` inside the repo.

## Next steps
- Try LightGBM / XGBoost and hyperparameter tuning.
- Add unit tests for preprocessing functions.
