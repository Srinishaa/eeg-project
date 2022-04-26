from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import models.eeg_rndforest

app = Flask(__name__)
model = pickle.load(open('eeg.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html', data = None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # FileStorage object wrapper
        file = request.files["file"]                    
        if file:
            df = pd.read_csv(file)

    label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    label_unmapping = {0 :'NEGATIVE', 1: 'NEUTRAL', 2 :'POSITIVE'}
    df['label'] = df['label'].replace(label_mapping)
    
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    predictions = model.predict(X)
    print(type(predictions))

    for p in predictions:
        print(label_unmapping.get(p))

    pred = label_unmapping.get(predictions[0])
    
    return render_template('index.html', data = pred)

if __name__ == "__main__":
    app.run(debug=True)