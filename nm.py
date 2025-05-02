# fraud_detection.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load dataset (download from: https://www.kaggle.com/mlg-ulb/creditcardfraud)
df = pd.read_csv("creditcard.csv")

# Check class distribution
print("Fraudulent vs Normal Transactions:")
print(df['Class'].value_counts())

# Handle imbalanced data (optional: undersample or use SMOTE)
fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0].sample(n=fraud.shape[0])  # undersample normal
balanced_df = pd.concat([fraud, normal])

# Features and labels
X = balanced_df.drop(['Class', 'Time'], axis=1)
y = balanced_df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Feature Importance Plot
plt.figure(figsize=(12, 6))
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Important Features")
plt.show()
