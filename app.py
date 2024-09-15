from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained models
disease_model = joblib.load('model_disease.pkl')
severity_model = joblib.load('model_severity.pkl')
treatment_model = joblib.load('model_treatment.pkl')

# Sample symptoms encoder (you'll need to load your encoder similarly)
encoder = joblib.load('mlb.pkl')

@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the NHealth API!'})

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    try:
        data = request.json
        symptoms = data.get('symptoms', [])

        # Encode the symptoms
        encoded_symptoms = encoder.transform([symptoms])

        # Predict the disease, severity, and treatment
        disease_prediction = disease_model.predict(encoded_symptoms)
        severity_prediction = severity_model.predict(encoded_symptoms)
        treatment_prediction = treatment_model.predict(encoded_symptoms)

        return jsonify({
            'disease': disease_prediction[0],
            'severity': severity_prediction[0],
            'treatment': treatment_prediction[0]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
