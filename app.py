from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open(r"C:\Users\Bhoomika T\OneDrive\Desktop\Randomforest\crop_recommendation.pkl", "rb"))
scaler = pickle.load(open(r"C:\Users\Bhoomika T\OneDrive\Desktop\Randomforest\scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Prepare and scale input
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)

        # Predict crop
        predicted_crop = model.predict(input_scaled)[0]

        return render_template('index.html',
                               crop=predicted_crop,
                               N=N, P=P, K=K, temperature=temperature,
                               humidity=humidity, ph=ph, rainfall=rainfall)

    except Exception as e:
        return render_template('index.html', crop=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
