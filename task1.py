# =========================
# Student Score Prediction
# =========================

# Data handling
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------
# Step 1: Create dataset
# --------------------------
data = {
    "StudyHours": [1,2,3,4,5,6,7,8,9,10],
    "ExamScore":  [35,40,50,55,60,65,70,75,85,90]
}
df = pd.DataFrame(data)
print("\nDataset:\n", df)

# --------------------------
# Step 2: Features & Target
# --------------------------
X = df[["StudyHours"]]   # Feature (independent variable)
y = df["ExamScore"]      # Target (dependent variable)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# Step 3: Train Model
# --------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Model parameters
print("\nModel Parameters:")
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)

# --------------------------
# Step 4: Predictions
# --------------------------
y_pred = model.predict(X_test)

print("\nPredictions vs Actual:")
print("Predicted:", y_pred)
print("Actual:", y_test.values)

# --------------------------
# Step 5: Evaluation
# --------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# --------------------------
# Step 6: Visualizations
# --------------------------
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# (A) Scatterplot with regression line
sns.scatterplot(ax=axes[0], x="StudyHours", y="ExamScore", data=df, color="blue", label="Actual Data")
axes[0].plot(X, model.predict(X), color="red", linewidth=2, label="Regression Line")
axes[0].set_title("StudyHours vs ExamScore")
axes[0].legend()

# (B) Actual vs Predicted
axes[1].scatter(y_test, y_pred, color="orange")
axes[1].set_xlabel("Actual Scores")
axes[1].set_ylabel("Predicted Scores")
axes[1].set_title("Actual vs Predicted Scores")

plt.tight_layout()
plt.show()
