import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

def load_data(file_path):
    """Load and preprocess the dataset."""
    loan_data = pd.read_csv(file_path)
    loan_data.columns = loan_data.columns.str.strip()  # Clean column names
    
    # Check for missing values
    if loan_data.isnull().sum().any():
        loan_data.fillna(method='ffill', inplace=True)  # Example to handle missing values
    
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    
    # Encode categorical variables
    loan_data['education'] = label_encoder.fit_transform(loan_data['education'])
    loan_data['self_employed'] = label_encoder.fit_transform(loan_data['self_employed'])
    loan_data['loan_status'] = label_encoder.fit_transform(loan_data['loan_status'])
    
    return loan_data

def train_model(data):
    """Train a Logistic Regression model."""
    X = data[['income_annum', 'cibil_score']]
    y = data['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    y_pred = logreg.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred))  # More detailed metrics
    
    return logreg

def save_model(model, filename):
    """Save the trained model to a file."""
    with open(filename, 'wb') as model_file:
        pickle.dump(model, model_file)

def predict(model, income, cibil_score):
    """Predict loan status for new input."""
    new_input = pd.DataFrame({'income_annum': [income], 'cibil_score': [cibil_score]})
    prediction = model.predict(new_input)[0]
    return "Approved" if prediction == 0 else "Rejected"

if __name__ == "__main__":
    file_path = 'loan_approval_dataset.csv'
    loan_data = load_data(file_path)
    
    model = train_model(loan_data)
    
    save_model(model, 'logistic_model.pkl')
    
    # Example prediction for a new input
    loan_status = predict(model, 500000, 750)
    print(f"Loan Status for new input: {loan_status}")
