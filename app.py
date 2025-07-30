from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('CKD_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        input_features = [float(request.form.get(f)) for f in request.form]
        prediction = model.predict([input_features])[0]

        if prediction == 0:
            result = "⚠️ Chronic Kidney Disease Detected. Please consult a nephrologist immediately."
        else:
            result = "✅ No signs of Chronic Kidney Disease. Keep maintaining a healthy lifestyle!"

        return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
