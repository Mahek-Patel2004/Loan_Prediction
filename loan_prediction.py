import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load the dataset
file_path = 'loan_approval_dataset.csv'  # Update the file path if necessary
loan_data = pd.read_csv(file_path)

# Clean column names by stripping any leading/trailing spaces
loan_data.columns = loan_data.columns.str.strip()

# Encode categorical variables: 'education', 'self_employed', and the target 'loan_status'
loan_data['education'] = loan_data['education'].map({'Graduate': 1, 'Not Graduate': 0})
loan_data['self_employed'] = loan_data['self_employed'].map({'Yes': 1, 'No': 0})
loan_data['loan_status'] = loan_data['loan_status'].map({'Approved': 1, 'Rejected': 0})

# Drop rows where the target variable 'loan_status' is NaN
loan_data.dropna(subset=['loan_status'], inplace=True)

# Separate features (X) and target (y)
X = loan_data.drop(columns=['loan_id', 'loan_status'])
y = loan_data['loan_status']

# Handle missing values in features using SimpleImputer (median imputation)
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)  # Impute missing values in the feature set

# Split data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the logistic regression model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Predict loan approval on the test set
y_pred = log_reg.predict(X_test)

# Evaluate the model using accuracy score and classification report
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print evaluation results
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:')
print(classification_rep)
