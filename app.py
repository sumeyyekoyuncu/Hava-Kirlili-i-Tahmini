from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
import pickle

app = Flask(__name__, template_folder='C:\\Users\\CASPER\\Desktop\\Veri Madenciliği\\veri madenciliği\\templates')
CORS(app)

DATA_PATH = 'global_air_pollution_dataset.csv'
MODEL_PATH = 'C:/Users/CASPER/Desktop/Veri Madenciliği/model.pkl'

with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api')
def api_home():
    return "🌍 Global Air Pollution REST API çalışıyor!"

def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    else:
        # Eğer veri dosyası yoksa boş dataframe dön
        return pd.DataFrame(columns=['Country', 'City', 'AQI Value', 'AQI Category', 'PM2.5 AQI Value', 'CO AQI Value', 'NO2 AQI Value'])

@app.route('/api/countries', methods=['GET'])
def get_countries():
    df = load_data()
    countries = df['Country'].dropna().unique().tolist()
    return jsonify(countries)

@app.route('/api/countries/<country>/cities', methods=['GET'])
def get_cities_by_country(country):
    df = load_data()
    cities = df[df['Country'].str.lower() == country.lower()]['City'].dropna().unique().tolist()
    if not cities:
        return jsonify({'error': 'Şehir bulunamadı'}), 404
    return jsonify(cities)

@app.route('/api/cities/<city_name>', methods=['GET'])
def get_city(city_name):
    df = load_data()
    city_data = df[df['City'].str.lower() == city_name.lower()]
    if city_data.empty:
        return jsonify({'error': 'Şehir bulunamadı'}), 404
    return jsonify(city_data.to_dict(orient='records'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        PM25 = data.get('PM25')
        CO = data.get('CO')
        NO2 = data.get('NO2')
        Ozone = data.get('Ozone')

        if PM25 is None or CO is None or NO2 is None or Ozone is None:
            return jsonify({'error': 'PM25, CO, NO2 ve Ozone değerleri eksik'}), 400

        features = np.array([PM25, CO, NO2, Ozone]).reshape(1, -1)
        prediction = model.predict(features)

        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
