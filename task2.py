# =========================================
# Task 2: Customer Segmentation (Clustering)
# Internship ML Task
# =========================================

# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------
# Step 1: Load Dataset
# --------------------------
# If you have the Kaggle dataset:
# df = pd.read_csv("data/Mall_Customers.csv")

# For now, we create a small demo dataset
data = {
    "CustomerID": range(1, 16),
    "AnnualIncome": [15,16,17,18,19,20,70,72,71,75,76,77,150,155,160],
    "SpendingScore": [39,81,6,77,40,76,6,8,5,55,80,90,15,20,10]
}
df = pd.DataFrame(data)
print("\nDataset Sample:\n", df.head())

# --------------------------
# Step 2: Feature Selection
# --------------------------
X = df[["AnnualIncome", "SpendingScore"]]

# Scale data (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# Step 3: Elbow Method
# --------------------------
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K, inertia, "bo-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.show()

# --------------------------
# Step 4: Apply K-Means
# --------------------------
# From elbow plot, let's pick k=3 (you can adjust)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Assignments:\n", df.head())

# --------------------------
# Step 5: Analyze Clusters
# --------------------------
print("\nCluster Summary (Mean values):")
print(df.groupby("Cluster")[["AnnualIncome", "SpendingScore"]].mean())

# --------------------------
# Step 6: Visualize Clusters
# --------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(x="AnnualIncome", y="SpendingScore",
                hue="Cluster", data=df, palette="Set1", s=100)

# Plot centroids
centers = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers)  # back to original scale
plt.scatter(centers[:,0], centers[:,1], c="black", s=200, alpha=0.7, marker="X", label="Centroids")

plt.title("Customer Segments (K-Means Clustering)")
plt.legend()
plt.show()
