# task3_fast.py
# Forest Cover Type Classification - Random Forest vs XGBoost

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# -----------------------------
# 1. Load Dataset (UCI Covertype)
# -----------------------------
print("Loading dataset...")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"

columns = [
    "Elevation", "Aspect", "Slope",
    "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon",
    "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"
] + [f"Wilderness_Area_{i}" for i in range(4)] + \
    [f"Soil_Type_{i}" for i in range(40)] + ["Cover_Type"]

# ✅ FIX: tell pandas it's gzip compressed
df = pd.read_csv(url, header=None, names=columns, compression="gzip")

print(f"Dataset shape: {df.shape}")

# -----------------------------
# 2. Reduce size for FAST testing
# -----------------------------
df = df.sample(20000, random_state=42)  # use only 20k rows for speed

X = df.drop("Cover_Type", axis=1)
y = df["Cover_Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# -----------------------------
# 3. Random Forest Classifier
# -----------------------------
print("\nTraining Random Forest...")

rf = RandomForestClassifier(random_state=42, n_jobs=-1)

rf_param_dist = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5]
}

rf_search = RandomizedSearchCV(
    rf,
    param_distributions=rf_param_dist,
    n_iter=4,
    scoring="accuracy",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_search.fit(X_train, y_train)

print("Best RF Parameters:", rf_search.best_params_)
print("Best RF CV Accuracy:", rf_search.best_score_)

rf_best = rf_search.best_estimator_
rf_pred = rf_best.predict(X_test)

print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_pred))

# Confusion Matrix RF
cm_rf = confusion_matrix(y_test, rf_pred)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf_best.classes_)
disp_rf.plot(cmap="Blues", xticks_rotation=45)
plt.title("Random Forest - Confusion Matrix")
plt.savefig("rf_confusion_matrix.png")
plt.close()
print("Saved: rf_confusion_matrix.png")

# -----------------------------
# 4. XGBoost Classifier
# -----------------------------
print("\nTraining XGBoost...")

# Shift labels for XGB (0–6 instead of 1–7)
y_train_xgb = y_train - 1
y_test_xgb = y_test - 1

xgb_clf = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=7,   # 7 classes (0–6)
    eval_metric="mlogloss",
    use_label_encoder=False,
    tree_method="hist",  # fast training
    n_jobs=-1
)

xgb_param_dist = {
    "max_depth": [3, 5],
    "learning_rate": [0.1, 0.3],
    "n_estimators": [50, 100],
    "subsample": [0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    xgb_clf,
    param_distributions=xgb_param_dist,
    n_iter=4,
    scoring="accuracy",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

xgb_search.fit(X_train, y_train_xgb)

print("Best XGB Parameters:", xgb_search.best_params_)
print("Best XGB CV Accuracy:", xgb_search.best_score_)

xgb_best = xgb_search.best_estimator_
xgb_pred = xgb_best.predict(X_test)

print("\nXGBoost Classification Report:\n", classification_report(y_test_xgb, xgb_pred))

# Confusion Matrix XGB
cm_xgb = confusion_matrix(y_test_xgb, xgb_pred)
disp_xgb = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=range(1, 8))
disp_xgb.plot(cmap="Greens", xticks_rotation=45)
plt.title("XGBoost - Confusion Matrix")
plt.savefig("xgb_confusion_matrix.png")
plt.close()
print("Saved: xgb_confusion_matrix.png")

# Feature Importance
xgb.plot_importance(xgb_best, max_num_features=10, importance_type="weight")
plt.title("Top 10 Important Features - XGBoost")
plt.savefig("xgb_feature_importance.png")
plt.close()
print("Saved: xgb_feature_importance.png")

print("\n✅ Done! Check saved PNG files for plots.")
