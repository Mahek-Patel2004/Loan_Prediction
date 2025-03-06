from flask import Flask, request, render_template
import pandas as pd
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained logistic regression model
with open('logistic_model.pkl', 'rb') as model_file:
    logistic_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        income_annum = float(request.form['income_annum'])
        cibil_score = float(request.form['cibil_score'])

        # Prepare input data for the model
        input_data = pd.DataFrame([[income_annum, cibil_score]], columns=['income_annum', 'cibil_score'])

        # Make the prediction
        prediction = logistic_model.predict(input_data)[0]

        # Determine loan status
        loan_status = "Approved" if prediction == 0 else "Rejected"

        # Render the result back to the HTML page with input values retained
        return render_template('index.html', prediction_text=f"Loan Status: {loan_status}",
                               income_annum=income_annum, cibil_score=cibil_score)

    except Exception as e:
        # Log the actual error message for debugging
        error_message = str(e)
        print(f"Error: {error_message}")  # Print the error to the console
        return render_template('index.html', prediction_text=f"Error occurred: {error_message}",
                               income_annum=request.form.get('income_annum', ''),
                               cibil_score=request.form.get('cibil_score', ''))

if __name__ == "__main__":
    app.run(debug=True)
