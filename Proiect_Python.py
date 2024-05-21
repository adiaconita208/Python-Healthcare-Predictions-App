# Import necessary libraries
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def read_csv_to_numpy_array(filename):
    data = []
    with open(filename, newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        ok = 0
        for row in csvreader:
            # Convert each row to a list of floats
            if (ok == 1):
                data.append([float(value) for value in row])
            else:
                ok = 1
    return np.array(data)

# Load your dataset (replace with your own data)
# Assume you have a CSV file with features and labels
data = pd.read_csv("/home/mihnea/Desktop/healthcare_data.csv")

# Assume the target column is named "disease" (1 for positive, 0 for negative)
X = data.drop(columns=["disease"])
y = data["disease"]

# Handle missing values if any
X = X.fillna(X.mean())

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression(solver='liblinear')  # Adding solver to handle convergence issues

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Generate a classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Function to transform new patient features with the same scaler
def transform_new_patient_features(scaler, feature_names, patient_features):
    patient_df = pd.DataFrame(patient_features, columns=feature_names)
    return scaler.transform(patient_df)

# Extract feature names
feature_names = data.columns.drop("disease")

# Load and predict on multiple patients from a CSV file
test_patients = read_csv_to_numpy_array("/home/mihnea/Desktop/patients.csv")

count = 0
with open('/home/mihnea/Desktop/disease_predictions.txt', 'w') as f:
    for row in test_patients:
        patient_features = np.array([row])
        patient_features = transform_new_patient_features(scaler, feature_names, patient_features)
        predicted_disease = model.predict(patient_features)
        print(f"Predicted disease status for patient {count}: {'Positive' if predicted_disease[0] == 1 else 'Negative'}", file=f)
        count += 1
