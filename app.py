from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import eeg

app = Flask(__name__)
model = pickle.load(open('eeg.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # FileStorage object wrapper
        file = request.files["file"]                    
        if file:
            df = pd.read_csv(file)
    print(df.head())

    label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
    df['label'] = df['label'].replace(label_mapping)
    
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    pred = model.predict(X)
    print(pred)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)