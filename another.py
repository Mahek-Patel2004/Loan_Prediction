import pandas as pd
import pickle

# Load the saved model
with open('logistic_model.pkl', 'rb') as model_file:
    logistic_model = pickle.load(model_file)

# Prepare new input for prediction
new_input = pd.DataFrame({
    'income_annum': [600000],
    'cibil_score': [700]
})

# Make the prediction
prediction = logistic_model.predict(new_input)[0]
loan_status = "Approved" if prediction == 0 else "Rejected"
print(f"Loan Status for new input: {loan_status}")
