import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Instead of SMOTE or ADASYN, let's try a different oversampling method (e.g., SVM-SMOTE):
from imblearn.over_sampling import SVMSMOTE

# Load the dataset
df = pd.read_csv(r"C:\Users\Nikhil Sukthe\Downloads\train_hsbc_df.csv")

# Define feature matrix X and target variable y
X = df.drop(columns=["fraud"])
y = df["fraud"]

# Convert 'age' column to numeric
X["age"] = pd.to_numeric(X["age"], errors='coerce')

# Convert 'gender' column to numeric (M -> 0, F -> 1)
X["gender"] = X["gender"].map({'M': 0, 'F': 1})

# Fill NaN values with 0
X = X.fillna(0)

# Select only numeric columns for scaling
numeric_cols = X.select_dtypes(include=['number']).columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numeric_cols])
X_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Use SVM-SMOTE to oversample the minority class
svm_smote = SVMSMOTE(random_state=42)
X_train_resampled, y_train_resampled = svm_smote.fit_resample(X_train, y_train)

# Build Neural Network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_resampled, y_train_resampled,
          epochs=20, batch_size=32,
          validation_data=(X_test, y_test))

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()

