from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all input values from form
        features = [float(x) for x in request.form.values()]

        # Convert to numpy array
        final_input = np.array([features])

        # Scale input
        final_input = scaler.transform(final_input)

        # Prediction
        prediction = model.predict(final_input)

        return render_template('index.html',
                               prediction_text=f"Predicted Room Occupancy: {prediction[0]}")

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)