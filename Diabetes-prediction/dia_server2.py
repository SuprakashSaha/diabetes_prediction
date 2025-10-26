from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load model
model = joblib.load('C:/Users/Admin/Desktop/Practice project/diabetis web app project/Diabetes-prediction/dia.pkl')

@app.route('/ridge/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    
    # Return JSON response with correct format
    return jsonify({"prediction": prediction.tolist()[0]})  

if __name__ == '__main__':
    app.run(debug=True)
