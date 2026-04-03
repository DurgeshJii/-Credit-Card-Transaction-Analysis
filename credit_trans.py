# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ================================
# 2. LOAD DATA
# ================================
df = pd.read_csv('card_transdata.csv')

print("Dataset Shape:", df.shape)
print(df.head())

# ================================
# 3. DATA ANALYSIS (EDA)
# ================================
print("\nINFO:")
print(df.info())

print("\nSTATISTICS:")
print(df.describe())

print("\nNULL VALUES:")
print(df.isnull().sum())

# Fraud distribution
sns.countplot(x='fraud', data=df)
plt.title("Fraud vs Non-Fraud")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# ================================
# 4. FEATURE & TARGET
# ================================
X = df.drop('fraud', axis=1)
y = df['fraud']

# ================================
# 5. TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 6. FEATURE SCALING
# ================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================
# 7. MODEL 1: LOGISTIC REGRESSION
# ================================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d')
plt.title("LR Confusion Matrix")
plt.show()

# ================================
# 8. MODEL 2: RANDOM FOREST
# ================================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d')
plt.title("RF Confusion Matrix")
plt.show()

# ================================
# 9. FEATURE IMPORTANCE
# ================================
importances = rf.feature_importances_
feature_names = X.columns

feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop Features:\n", feat_df.head())

sns.barplot(x='Importance', y='Feature', data=feat_df.head(10))
plt.title("Top 10 Important Features")
plt.show()

# ================================
# 10. SAVE MODEL
# ================================
import joblib

joblib.dump(rf, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModel Saved Successfully!")