import numpy as np
import plotly.graph_objects as go
from joblib import load

import warnings

def predict_with_proba(X_test):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_test = X_test.reshape(1, -1)
        logistic_regression, scaler = load('logistic_regression.joblib')

        X_test_scaled = scaler.transform(X_test)
        predicted_probabilities = logistic_regression.predict_proba(X_test_scaled)
        predicted_classes = logistic_regression.predict(X_test_scaled)

        label_names = logistic_regression.classes_  # Get the label names
        # print(label_names)
        for i, (label, probabilities) in enumerate(zip(predicted_classes, predicted_probabilities)):
            label_name = label_names[label]  # Get the corresponding label name
            print(f"Instance {i + 1}: Predicted Class = {label_name}, Probabilities = {probabilities}")

        fig = go.Figure(data=[go.Pie(labels=label_names, values=probabilities, hole=0.4)])
        print(predicted_classes)
        print(label_names)
        fig.update_layout(title='Predicted Probabilities for Each Class')
        fig.show()
    return predicted_probabilities

X_new = np.array([1., 0.17021277, 0.9754268, 0.12622076, 0.5, 0.7, 0.5, 0.86666667, 0.66666667, 0., 0.6, 0., 0., 1., 0.66666667, 0.])
predict_with_proba(X_new)

