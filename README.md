# AI Lab - Data Preprocessing for ANN

NED University of Engineering & Technology
Lab Session 03 - Applying Data Preprocessing for ANN

## Dataset

Loan Approval dataset — 614 rows, 13 columns.
Target column: `Loan_Status` (`Y` = approved, `N` = rejected)

| Property | Value |
|---|---|
| Number of input parameters | 11 |
| Number of output categories | 2 (Y = 422, N = 192) |
| Rows with missing values | LoanAmount, Loan_Amount_Term, Credit_History, Gender, Dependents, Self_Employed |

## Preprocessing techniques applied

| Method | Applied to |
|---|---|
| Standardization | `LoanAmount`, `Loan_Amount_Term` |
| Min-max scaling | `ApplicantIncome`, `CoapplicantIncome`, `loan_to_income` |
| Binary scaling | `Credit_History` |
| One-hot encoding | `Gender`, `Married`, `Self_Employed`, `Property_Area` |
| Ordinal encoding + scaling | `Dependents`, `Education`, `income_bin` |
| Binning | Total income -> `low`, `middle`, `high`, `very_high` |
| Missing values | Median (numeric columns), most frequent (categorical columns) |

`loan_to_income` and `income_bin` are engineered features derived from the raw
columns before preprocessing (see `clean_and_engineer_features()` in the code).

## ANN model

A small `MLPClassifier` is trained on the preprocessed features to predict
loan approval:

- Hidden layers: `(18, 9)`
- Learning rate: `0.01` (adaptive)
- Momentum: `0.9`
- Solver: `sgd`
- Early stopping on a 15% validation split

## Files

- `preprocessing.py` — loads the dataset, applies all preprocessing steps,
  trains the ANN, and queries it with a sample applicant.
- `data/` — downloaded dataset is cached here at runtime (not committed).

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas scikit-learn
python3 preprocessing.py
```

## Sample query

| Gender | Married | Dependents | Education | Self_Employed | ApplicantIncome | CoapplicantIncome | LoanAmount | Loan_Amount_Term | Credit_History | Property_Area |
|---|---|---|---|---|---|---|---|---|---|---|
| Male | Yes | 1 | Graduate | No | 5200 | 1600 | 145 | 360 | 1 | Semiurban |

Predicted class: **1 (approved)** — approval probability ≈ **0.87**

## Author

Name: Syed Irfan Raza
Roll No: CS-24118
