from flask import Flask, request, url_for, redirect, render_template, jsonify, abort
import pandas as pd
import pickle
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    scale = StandardScaler()
    df = pd.read_csv("static/data/water_potability.csv")
    df = scale.fit(df.drop(columns="Potability", axis=1))
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    data_unseen_scaled = scale.transform(data_unseen)
    prediction = model.predict(data_unseen_scaled)
    prediction = int(prediction[0])
    # return render_template('index.html', pred=prediction)
    return json.dumps({'potable':prediction})

if __name__ == '__main__':
    app.run(debug=True)