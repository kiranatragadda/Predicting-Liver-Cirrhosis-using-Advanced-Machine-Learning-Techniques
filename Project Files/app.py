from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('rf_acc_68.pkl', 'rb'))
scaler = pickle.load(open('normalizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user inputs
        features = [float(x) for x in request.form.values()]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        result = "High Risk of Cirrhosis" if prediction == 1 else "Low Risk of Cirrhosis"
        return render_template('inner-page.html', prediction_text=result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
