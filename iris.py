# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA

# Set style for plots
sns.set_theme(style="whitegrid")

# ==========================================
# STEP 1: Load Custom Dataset (iris.txt)
# ==========================================
print("--- Step 1: Loading iris.txt Dataset ---")

# Load dataset using Pandas (since it's comma-separated)
df = pd.read_csv("D:\iris.txt")

# Display dataset info
print(f"Dataset Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Define Features (X) and Target (y)
# Features are the measurement columns, target is 'species'
X = df.drop(columns=['species'])
y = df['species']
target_names = y.unique() # ['setosa', 'versicolor', 'virginica']

# ==========================================
# STEP 2: Exploratory Data Analysis (Pairplot)
# ==========================================
print("\n--- Step 2: Generating Exploratory Pairplot ---")
sns.pairplot(df, hue='species', palette='Set2')
plt.suptitle("Pairplot of Iris Dataset Features", y=1.02)
plt.show()

# ==========================================
# STEP 3: Train / Test Split & Feature Scaling
# ==========================================
print("\n--- Step 3: Train/Test Split (80/20) & Normalization ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training Samples: {X_train_scaled.shape[0]}, Test Samples: {X_test_scaled.shape[0]}")

# ==========================================
# STEP 4: Train Models & Evaluate Metrics
# ==========================================
print("\n--- Step 4: Model Training and Benchmarking ---")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

predictions = {}

for name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    
    # Evaluate metrics (using 'weighted' average for multi-class classification)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n==================== {name} ====================")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# ==========================================
# STEP 5: Confusion Matrix Heatmaps
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred, labels=target_names)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_title(f'Confusion Matrix: {name}')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.show()

# ==========================================
# STEP 6: Decision Boundary Visualization (2D PCA)
# ==========================================
print("\n--- Step 6: Visualizing Decision Boundaries ---")

# Reduce features to 2D using PCA for visualization purposes
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Convert string labels to numbers (0, 1, 2) for boundary plotting color maps
y_train_encoded = pd.factorize(y_train)[0]
y_test_encoded = pd.factorize(y_test)[0]

def plot_decision_boundary(model, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # If predictions are strings, map them back to numeric codes for plotting
    if Z.dtype.kind in 'USO':
        Z = pd.factorize(Z)[0]
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=40)
    ax.set_title(title)
    ax.set_xlabel('PCA Feature 1')
    ax.set_ylabel('PCA Feature 2')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (name, model_class) in zip(axes, [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42))
]):
    clf_2d = model_class
    clf_2d.fit(X_train_pca, y_train)
    plot_decision_boundary(clf_2d, X_test_pca, y_test_encoded, f"{name} Boundary (PCA 2D)", ax)

plt.tight_layout()
plt.show()