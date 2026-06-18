# ============================================
# IRIS FLOWER CLASSIFICATION - CodeAlpha Task
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ─────────────────────────────────────────────
# 1. LOAD THE DATASET
# ─────────────────────────────────────────────
iris = load_iris()

# Convert to a DataFrame for easy viewing
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
})

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)
print(df.head(10))
print(f"\nShape: {df.shape}")
print(f"\nClass Distribution:\n{df['species_name'].value_counts()}")

# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────

# Plot 1: Pairplot to see relationships
sns.pairplot(df, hue='species_name', palette='Set2',
             vars=iris.feature_names)
plt.suptitle('Iris Feature Relationships', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig('pairplot.png')
plt.show()
print("\n✅ Pairplot saved as pairplot.png")

# Plot 2: Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(df[iris.feature_names].corr(),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('heatmap.png')
plt.show()
print("✅ Heatmap saved as heatmap.png")

# ─────────────────────────────────────────────
# 3. PREPARE DATA FOR TRAINING
# ─────────────────────────────────────────────
X = df[iris.feature_names]   # Features (input)
y = df['species']             # Labels (output)

# Split: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 50)
print("DATA SPLIT")
print("=" * 50)
print(f"Training samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

# ─────────────────────────────────────────────
# 4. TRAIN THE MODEL
# ─────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\n✅ Model trained successfully!")

# ─────────────────────────────────────────────
# 5. EVALUATE THE MODEL
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=iris.target_names
))

# Plot 3: Confusion Matrix
plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("✅ Confusion matrix saved as confusion_matrix.png")

# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
importances = pd.Series(
    model.feature_importances_,
    index=iris.feature_names
).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index,
            palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
print("✅ Feature importance chart saved")

# ─────────────────────────────────────────────
# 7. PREDICT ON NEW FLOWER (Demo)
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("PREDICT A NEW FLOWER")
print("=" * 50)

new_flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                          columns=iris.feature_names)
prediction = model.predict(new_flower)
species = iris.target_names[prediction[0]]

print(f"Input measurements : {new_flower.values.tolist()[0]}")
print(f"Predicted species  : ✅ {species.upper()}")