from flask import Flask, request, jsonify
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the models and vectorizer
model_disease = joblib.load('model_disease.pkl')
model_severity = joblib.load('model_severity.pkl')
model_treatment = joblib.load('model_treatment.pkl')
vectorizer = joblib.load('symptom_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    
    if 'symptoms' not in data:
        return jsonify({"error": "Symptom data is required"}), 400
    
    symptoms = data['symptoms']
    
    # Transform the input symptoms into the vectorized form
    symptoms_vectorized = vectorizer.transform([symptoms])
    
    # Make predictions using the loaded models
    predicted_disease = model_disease.predict(symptoms_vectorized)[0]
    predicted_severity = model_severity.predict(symptoms_vectorized)[0]
    predicted_treatment = model_treatment.predict(symptoms_vectorized)[0]
    
    # Return the predictions as a JSON response
    return jsonify({
        'disease': predicted_disease,
        'severity': predicted_severity,
        'treatment': predicted_treatment
    })

if __name__ == '__main__':
    app.run(debug=True)
