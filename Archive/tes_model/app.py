from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scaler
model = load_model('food_recommendation_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

from flask import render_template

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Get input data from request
        data = request.get_json()
        
        # Proses input pengguna
        features = np.array([
            data['calories'],
            data['protein_content'],
            data['carbohydrate_content'],
            data['cook_time']
        ]).reshape(1, -1)

        # Normalize input features
        features = scaler.transform(features)

        # Use model to predict
        score = model.predict(features)

        return jsonify({'score': score[0][0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)