import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('static/emotions.csv')

label_mapping = {'NEGATIVE': 0, 'NEUTRAL': 1, 'POSITIVE': 2}
def preprocess_inputs(df):
    df = df.copy()
    
    df['label'] = df['label'].replace(label_mapping)
    
    y = df['label'].copy()
    X = df.drop('label', axis=1).copy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(data)

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(999, inplace=True)
y_train.replace([np.inf, -np.inf], np.nan, inplace=True)
y_train.fillna(999, inplace=True)

clf = RandomForestClassifier(n_estimators = 78)
clf.fit(X_train, y_train)

pickle.dump(clf, open('eeg.pkl', 'wb'))
