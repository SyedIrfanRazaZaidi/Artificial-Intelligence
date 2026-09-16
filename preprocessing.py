"""
AI Lab - Lab Session 03
Applying Data Preprocessing for ANN

NED University of Engineering & Technology
Dataset: Loan Approval dataset (614 rows, 13 columns)
Target column: Loan_Status (Y = approved, N = rejected)
"""

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# ---------------------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------------------
CSV_URL = "https://gist.githubusercontent.com/lego963/c1edf566360f644c7bc1a107c9fa0894/raw/loan.csv"
CSV_PATH = Path("data/loan.csv")

CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
if not CSV_PATH.exists():
    urllib.request.urlretrieve(CSV_URL, CSV_PATH)

raw_data = pd.read_csv(CSV_PATH)
raw_data.columns = [str(column).strip() for column in raw_data.columns]

print("Dataset loaded from:", CSV_PATH)
print("Shape:", raw_data.shape)
print(raw_data.head(8))

# ---------------------------------------------------------------------------
# 2. Dataset description (Exercise 1 & 2)
# ---------------------------------------------------------------------------
numeric_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
categorical_columns = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
target_column = "Loan_Status"

required_columns = set(numeric_columns + categorical_columns + [target_column])
missing_columns = sorted(required_columns - set(raw_data.columns))
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("Number of input parameters:", len(numeric_columns + categorical_columns))
print("Number of output categories:", raw_data[target_column].nunique())
print("Output category counts:")
print(raw_data[target_column].value_counts())
print("\nMissing values per column:")
print(raw_data[numeric_columns + categorical_columns + [target_column]].isna().sum())

# ---------------------------------------------------------------------------
# 3. Feature engineering + preprocessing techniques (Exercise 3 & 4)
# ---------------------------------------------------------------------------
# Preprocessing methods applied:
#   Standardization        -> LoanAmount, Loan_Amount_Term
#   Min-max scaling         -> ApplicantIncome, CoapplicantIncome, loan_to_income
#   Binary scaling          -> Credit_History
#   One-hot encoding        -> Gender, Married, Self_Employed, Property_Area
#   Ordinal encoding+scaling-> Dependents, Education, income_bin
#   Binning                 -> Total income -> low, middle, high, very_high
#   Missing values          -> Median for numeric, most frequent for categorical


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the extra ordinal/numeric features used by the preprocessor."""
    df = df.copy()

    # Dependents comes in as "0", "1", "2", "3+" -> normalize the "3+" label
    df["Dependents"] = df["Dependents"].astype(str).str.replace("+", "", regex=False)
    df.loc[~df["Dependents"].isin(["0", "1", "2", "3"]), "Dependents"] = df["Dependents"]

    # Derived numeric features
    total_income = df["ApplicantIncome"].fillna(0) + df["CoapplicantIncome"].fillna(0)
    df["Total_Income"] = total_income
    df["loan_to_income"] = df["LoanAmount"] / total_income.replace(0, 1)

    # Binning: Total income -> low, middle, high, very_high
    df["income_bin"] = pd.cut(
        df["Total_Income"],
        bins=[-float("inf"), 4000, 7000, 12000, float("inf")],
        labels=["low", "middle", "high", "very_high"],
    ).astype(str)

    return df


engineered_data = clean_and_engineer_features(raw_data)

standard_cols = ["LoanAmount", "Loan_Amount_Term"]
minmax_cols = ["ApplicantIncome", "CoapplicantIncome", "loan_to_income"]
binary_cols = ["Credit_History"]
nominal_cols = ["Gender", "Married", "Self_Employed", "Property_Area"]
ordinal_cols = ["Dependents", "Education", "income_bin"]

feature_columns = standard_cols + minmax_cols + binary_cols + nominal_cols + ordinal_cols

ordinal_categories = [
    ["0", "1", "2", "3"],                       # Dependents
    ["Not Graduate", "Graduate"],                # Education
    ["low", "middle", "high", "very_high"],      # income_bin
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "standard",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            standard_cols,
        ),
        (
            "minmax",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", MinMaxScaler()),
            ]),
            minmax_cols,
        ),
        (
            "binary",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
            ]),
            binary_cols,
        ),
        (
            "nominal",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]),
            nominal_cols,
        ),
        (
            "ordinal",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(categories=ordinal_categories)),
                ("scale", MinMaxScaler()),
            ]),
            ordinal_cols,
        ),
    ]
)

# Preview the preprocessed data (Exercise 4)
preview_matrix = preprocessor.fit_transform(engineered_data[feature_columns])
preview_columns = (
    [f"standard__{c}" for c in standard_cols]
    + [f"minmax__{c}" for c in minmax_cols]
    + [f"binary__{c}" for c in binary_cols]
    + list(preprocessor.named_transformers_["nominal"]["encode"].get_feature_names_out(nominal_cols))
    + [f"ordinal__{c}" for c in ordinal_cols]
)
preprocessed_preview = pd.DataFrame(preview_matrix, columns=preview_columns)
print("\nPreprocessed data preview:")
print(preprocessed_preview.head(5).T.round(2))

# ---------------------------------------------------------------------------
# 4. Train an ANN (MLPClassifier) on the preprocessed data
# ---------------------------------------------------------------------------
x = engineered_data[feature_columns]
y = engineered_data[target_column].map({"Y": 1, "N": 0})

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

learning_rate = 0.01
momentum = 0.9
hidden_layers = (18, 9)

model = Pipeline([
    ("preprocess", preprocessor),
    (
        "ann",
        MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="sgd",
            learning_rate="adaptive",
            learning_rate_init=learning_rate,
            momentum=momentum,
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=40,
            random_state=42,
        ),
    ),
])

model.fit(x_train, y_train)
predictions = model.predict(x_test)

ann = model.named_steps["ann"]
accuracy = accuracy_score(y_test, predictions)
input_neurons = model.named_steps["preprocess"].transform(x_train.iloc[:1]).shape[1]

print("\nNumber of output categories:", y.nunique())
print("Number of data rows:", len(x))
print("Number of training rows:", len(x_train))
print("Number of testing rows:", len(x_test))
print("Learning rate:", learning_rate)
print("Momentum:", momentum)
print("Number of layers: 4 (input + 2 hidden + output)")
print(f"Size of layers: input={input_neurons}, hidden={hidden_layers}, output=1")
print("Number of cycles for training:", ann.n_iter_)
print(f"Percentage of correctness in results: {accuracy * 100:.2f}%")
print(f"Final training loss: {ann.loss_:.6f}")

# ---------------------------------------------------------------------------
# 5. Query the trained ANN with a new applicant
# ---------------------------------------------------------------------------
query_raw = pd.DataFrame([
    {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5200,
        "CoapplicantIncome": 1600,
        "LoanAmount": 145,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Semiurban",
    }
])

query = clean_and_engineer_features(query_raw)[feature_columns]
predicted_class = int(model.predict(query)[0])
approval_probability = float(model.predict_proba(query)[0, 1])

print("\nQuery:")
print(query_raw)
print("Predicted class:", predicted_class, "(1 = approved, 0 = rejected)")
print(f"Approval probability: {approval_probability:.4f}")
