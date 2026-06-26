# ============================================
# Telco Customer Churn Prediction Model
# Part 1: Import, Data Cleaning & Preprocessing
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

# ============================================
# Load Dataset
# ============================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values Before Cleaning")
print(df.isnull().sum())

# ============================================
# Data Cleaning
# ============================================

print("\nCleaning Dataset...")

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Fill missing values with median
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# Remove Customer ID
df.drop(columns=["customerID"], inplace=True)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

print("\nTotal Missing Values Remaining:")
print(df.isnull().sum().sum())

# Save cleaned dataset (optional but useful)
df.to_csv(
    "data/cleaned_telco.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")

# ============================================
# Encode Categorical Columns
# ============================================

print("\nEncoding Categorical Features...")

label_encoders = {}

categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    label_encoders[column] = encoder

print("\nEncoded Columns")

print(list(categorical_columns))

# Save encoders
joblib.dump(
    label_encoders,
    "models/label_encoders.pkl"
)

print("Label Encoders Saved")

# ============================================
# Separate Features and Target
# ============================================

X = df.drop("Churn", axis=1)

y = df["Churn"]

print("\nFeature Shape:", X.shape)

print("Target Shape:", y.shape)

# ============================================
# Save Feature Columns
# ============================================

feature_columns = X.columns.tolist()

joblib.dump(
    feature_columns,
    "models/feature_columns.pkl"
)

print("\nFeature Columns Saved")

print(feature_columns)

# ============================================
# Final Missing Value Check
# ============================================

print("\nChecking Missing Values Before Training")

print(X.isnull().sum())

print("\nTotal Missing Values:", X.isnull().sum().sum())

# Stop execution if missing values exist
if X.isnull().sum().sum() > 0:

    raise ValueError(
        "Dataset still contains missing values."
    )

# ============================================
# Train-Test Split
# ============================================

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("\nTraining Data Shape")

print(X_train.shape)

print("\nTesting Data Shape")

print(X_test.shape)

print("\nData preprocessing completed successfully.")

print("=" * 60)
print("Ready for Model Training")
print("=" * 60)

# ============================================
# Model Training
# ============================================

print("\n" + "=" * 60)
print("Training Machine Learning Models...")
print("=" * 60)

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )

}

results = []

best_model = None
best_model_name = ""
best_accuracy = 0

# ============================================
# Train & Evaluate Each Model
# ============================================

for name, model in models.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Train
    model.fit(X_train, y_train)

    # Predict
    prediction = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    # Store Results
    results.append({

        "Model": name,

        "Accuracy": round(accuracy, 4),

        "Precision": round(precision, 4),

        "Recall": round(recall, 4),

        "F1 Score": round(f1, 4)

    })

    # Print Metrics
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Classification Report
    print("\nClassification Report\n")

    print(classification_report(y_test, prediction))

    # Confusion Matrix
    print("Confusion Matrix\n")

    print(confusion_matrix(y_test, prediction))

    # Save Best Model
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = name


# ============================================
# Model Comparison
# ============================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n" + "=" * 60)
print("Model Comparison")
print("=" * 60)

print(results_df)

# ============================================
# Save Model Comparison
# ============================================

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nModel comparison saved.")

# ============================================
# Best Model
# ============================================

print("\n" + "=" * 60)
print("Best Performing Model")
print("=" * 60)

print("Model    :", best_model_name)

print("Accuracy :", round(best_accuracy, 4))

# ============================================
# Save Best Model
# ============================================

joblib.dump(
    best_model,
    "models/churn_model.pkl"
)

print("\nBest model saved as")

print("models/churn_model.pkl")

# ============================================
# Save Training Summary
# ============================================

summary = {

    "Best Model": best_model_name,

    "Accuracy": round(best_accuracy, 4),

    "Training Samples": len(X_train),

    "Testing Samples": len(X_test),

    "Number of Features": len(feature_columns)

}

joblib.dump(
    summary,
    "models/model_summary.pkl"
)

print("\nTraining summary saved.")

# ============================================
# Training Completed
# ============================================

print("\n" + "=" * 60)
print("Training Completed Successfully")
print("=" * 60)

print("\nGenerated Files:")

print("✔ models/churn_model.pkl")

print("✔ models/label_encoders.pkl")

print("✔ models/feature_columns.pkl")

print("✔ models/model_comparison.csv")

print("✔ models/model_summary.pkl")

print("✔ data/cleaned_telco.csv")

print("\nProject is ready for Streamlit deployment.")