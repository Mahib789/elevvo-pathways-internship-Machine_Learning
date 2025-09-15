# task4.py
# Loan Approval Prediction using XGBoost + SMOTE (handles imbalance)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("train model.csv")

print("Dataset Loaded Successfully ✅")
print("First 5 rows:\n", df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# -----------------------------
# 2. Preprocessing
# -----------------------------
# Drop Loan_ID if present
if "Loan_ID" in df.columns:
    df = df.drop("Loan_ID", axis=1)

# Fill missing values
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include=["int64", "float64"]).columns:
    df[col] = df[col].fillna(df[col].median())

# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# -----------------------------
# 3. Split Data
# -----------------------------
X = df.drop("Loan_Status_Y", axis=1, errors="ignore")  # target is Loan_Status_Y after encoding
y = df["Loan_Status_Y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# -----------------------------
# 4. Handle Imbalance with SMOTE
# -----------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("\nAfter SMOTE Resampling:")
print("Class distribution in y_train_res:\n", y_train_res.value_counts())

# -----------------------------
# 5. Train Model
# -----------------------------
model = xgb.XGBClassifier(
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)

model.fit(X_train_res, y_train_res)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# 7. Feature Importance
# -----------------------------
xgb.plot_importance(model, max_num_features=10, importance_type="weight")
plt.title("Top 10 Feature Importances")
plt.show()
