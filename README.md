# VERY IMPORTANT:
- Use a Python 3.11 environment.

## Requirements

install dependencies:

```powershell
pip install -r requirements.txt
```

## Data Files

- `data/adult.csv`: cleaned Adult dataset used by both experiments
- `data/adult-domain.json`: discrete domain specification used by the DP generator

The datasets in this repo are already arranged for the experiments, so you do not need to download anything else to reproduce the current outputs.

## 1. Run Differential Privacy Synthetic Data Generation

Run it with:

```powershell
python src\rmckenna_vendor\match3.py --dataset data\adult.csv --specs data\adult-domain.json --epsilon 1.0 --delta 2.2820544e-12 --save outputs\synthetic-1.0.csv
```

What this does:

- reads the Adult dataset
- adds calibrated Gaussian noise
- compresses low-support categories
- learns a graphical model
- samples a synthetic dataset
- writes the result to `outputs/synthetic-1.0.csv`

Expected output:

- `outputs/synthetic-1.0.csv` in the outputs folder
- log messages showing the calibrated noise level and optimization progress

## 2. Run K-Anonymity Experiment

The K-anonymity workflow is in `notebooks/k_anonymity_experiment.ipynb`.

Open the notebook and run all cells from top to bottom.

What the notebook does:

- loads the Adult dataset
- cleans missing values
- applies Mondrian K-anonymity to several `k` values
- saves anonymized outputs into `outputs/`
- evaluates a Linear SVM on the original generalized data and each anonymized dataset
- generates accuracy / precision / recall / AUC plots

Outputs produced by the notebook:

- `outputs/adult_raw_generalized.csv`
- `outputs/adult_k2.csv`
- `outputs/adult_k5.csv`
- `outputs/adult_k10.csv`
- `outputs/adult_k20.csv`
- `outputs/adult_k50.csv`

### K values used

The notebook evaluates:

- `k = 2`
- `k = 5`
- `k = 10`
- `k = 20`
- `k = 50`

## 3. Compare K-Anonymity vs Differential Privacy

The comparison utility is in `src/privacy_compare.py`.

It summarizes:

- shared columns between two CSV files
- number of unique values per shared column
- missing-rate / top-value concentration
- label balance / positive rate

### Compare the existing outputs

If your Python 3.11 environment is already active, you can compare `adult_k2.csv` with `outputs/synthetic-1.0.csv` directly:

```powershell
python src\privacy_compare.py --left outputs\adult_k2.csv --right outputs\synthetic-1.0.csv --output-dir outputs\comparison_test
```

This writes:

- `outputs/comparison_test/shared_columns.csv`
- `outputs/comparison_test/label_balance.csv`

### Command-line usage

You can also use the same command with your activated environment:

```powershell
python src\privacy_compare.py --left outputs\adult_k2.csv --right outputs\synthetic-1.0.csv --output-dir outputs\comparison_test
```

## 4. What the Outputs Mean

### K-Anonymity outputs

Files such as `outputs/adult_k2.csv` contain generalized quasi-identifiers:

- numeric QIs become interval strings like `[25,28]`
- categorical QIs become set strings like `{Private|Self-emp}`
- numeric helper features are added as `<column>_mid` and `<column>_width`
- `eq_class_id` identifies the equivalence class
- `eq_class_size` shows how many records share that generalized pattern

### Differential privacy output

`outputs/synthetic-1.0.csv` is a fully synthetic dataset generated from noisy statistics.
It keeps the same schema as the original Adult data, but the values are synthetic integer codes rather than raw text labels.

## 5. Evaluation Metrics

The K-anonymity notebook evaluates a Linear SVM with these metrics:

- misclassification rate
- accuracy
- precision
- recall
- AUC

The comparison utility currently focuses on data-level summaries rather than model metrics.

## 6. Running Tests

The repo includes lightweight tests for the K-anonymity and comparison helpers.

If `pytest` is installed in your environment, run:

```powershell
& ".venv311\Scripts\python.exe" -m pytest -q
```

If `pytest` is not installed, install it first:

```powershell
& ".venv311\Scripts\python.exe" -m pip install pytest
```

## 7. Common Troubleshooting

### `ModuleNotFoundError: No module named 'mbi'`

Use the provided Python 3.11 environment and install dependencies with `pip install -r requirements.txt`.

### `ImportError` involving NumPy 2.x and pandas / pyarrow

This usually means the wrong Python interpreter is being used. Run the project with `.venv311\Scripts\python.exe` instead of system Python.

### `ValueError: Expected integer data, got object`

This was fixed in `src/rmckenna_vendor/match3.py`. If you modify the code and see it again, make sure transformed columns are cast to `int` before building an `mbi.Dataset`.

### `pytest` not found

Install it inside the project environment:

```powershell
& ".venv311\Scripts\python.exe" -m pip install pytest
```

## 8. Reproducibility Checklist

To reproduce the current project state from scratch:

1. Activate or create a Python 3.11 environment.
2. Install dependencies from `requirements.txt`.
3. Run `src/rmckenna_vendor/match3.py` to generate `outputs/synthetic-1.0.csv`.
4. Open `notebooks/k_anonymity_experiment.ipynb` and run all cells.
5. Run `src/privacy_compare.py` or the inline comparison command to generate comparison summaries.
6. Inspect the CSV outputs in `outputs/` and write the report.

## 9. Main Files

- `src/rmckenna_vendor/match3.py`: differential privacy synthetic data generator
- `src/rmckenna_vendor/mechanism.py`: shared data loading / output logic for the DP pipeline
- `src/k_anonymity/mondrian.py`: Mondrian-style K-anonymity implementation
- `src/k_anonymity/ml_eval.py`: evaluation helpers for the K-anonymity experiment
- `src/privacy_compare.py`: comparison utility for K-anonymity and DP outputs
- `notebooks/k_anonymity_experiment.ipynb`: end-to-end K-anonymity experiment notebook

## 10. Expected Project Workflow

For the full assignment, run the project in this order:

1. Generate the DP synthetic dataset with `match3.py`.
2. Run the K-anonymity notebook to create the anonymized datasets and model results.
3. Use `privacy_compare.py` to summarize differences between the K-anonymized data and the DP synthetic data.
4. Write the report using the generated CSVs and the model metrics from the notebook.

## License

This project is for coursework use.
