import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('PlayTennis.csv')

# Display the first few rows to verify loading (optional)
print(data.head())

# Prepare features (X) and target (y)
X = data.drop('Play Tennis', axis=1)
y = data['Play Tennis']

# Encode categorical features and target
label_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

le_y = LabelEncoder()
y = le_y.fit_transform(y)
label_encoders['Play Tennis'] = le_y

# Split the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define criteria to use
criteria = ['gini', 'entropy']

# Dictionary to store models and accuracies
models = {}
accuracies = {}

# Train Decision Tree models using different criteria
for criterion in criteria:
    # Create Decision Tree model with the specified criterion
    model = DecisionTreeClassifier(criterion=criterion)
    
    # Fit the model on training data
    model.fit(X_train, y_train)
    
    # Store the model
    models[criterion] = model
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    
    # Store accuracy
    accuracies[criterion] = acc

# Report and compare accuracies
print("Model Accuracies:")
for criterion, acc in accuracies.items():
    print(f"{criterion.capitalize()} criterion: {acc}")

# Print evaluation details for each criterion
for criterion in criteria:
    model = models[criterion]
    y_pred = model.predict(X_test)
    print(f"\nClassification Report (for {criterion.capitalize()} criterion):")
    print(classification_report(y_test, y_pred))
    print(f"Confusion Matrix (for {criterion.capitalize()} criterion):")
    print(confusion_matrix(y_test, y_pred))