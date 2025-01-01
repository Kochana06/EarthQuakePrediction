from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras import losses
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix

app = Flask(__name__)

# Load Models
rf_model_path = "models/random_forest_model.pkl"
svm_model_path = "models/svm_model.pkl"
xgb_model_path = "models/xgboost_model.pkl"
lstm_model_path = "models/lstm_model.h5"
transformer_model_path = "models/transformer_model.h5"

rf_model_loaded = joblib.load(rf_model_path)
svm_model_loaded = joblib.load(svm_model_path)
xgb_model_loaded = joblib.load(xgb_model_path)
lstm_model_loaded = load_model(lstm_model_path)
transformer_model_loaded = load_model(transformer_model_path, custom_objects={'mse': losses.MeanSquaredError()})

# Test Data and Ground Truth for Evaluation (Replace with your data)
X_test = np.random.rand(10, 16)  # Replace with real test data
y_test = np.random.randint(0, 2, 10)  # Replace with real labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get model selection and input data
    model_choice = request.form['model']
    data = request.form.to_dict()

    try:
        # Convert input values to float
        custom_input = np.array([[float(data[key]) for key in [
            'latitude', 'longitude', 'depth', 'mag', 'nst', 'gap',
            'dmin', 'rms', 'horizontalError', 'depthError', 'magError',
            'magNst', 'year', 'month', 'day', 'hour'
        ]]])
        
        # For simplicity, assuming no scaling is needed
        custom_input_scaled = custom_input

        prediction = None
        metrics = {}

        if model_choice == "rf":
            prediction = rf_model_loaded.predict(custom_input)
            y_pred = rf_model_loaded.predict(X_test)
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
                "Classification Report": classification_report(y_test, y_pred, output_dict=True)
            }

        elif model_choice == "svm":
            prediction = svm_model_loaded.predict(custom_input)
            y_pred = svm_model_loaded.predict(X_test)
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
                "Classification Report": classification_report(y_test, y_pred, output_dict=True)
            }

        elif model_choice == "xgb":
            prediction = xgb_model_loaded.predict(custom_input)
            y_pred = xgb_model_loaded.predict(X_test)
            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
                "Classification Report": classification_report(y_test, y_pred, output_dict=True)
            }

        elif model_choice == "lstm":
            custom_input_lstm = custom_input_scaled.reshape((1, 1, custom_input_scaled.shape[1]))
            prediction = lstm_model_loaded.predict(custom_input_lstm)
            y_pred = lstm_model_loaded.predict(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))
            metrics = {
                "MSE": mean_squared_error(y_test, y_pred)
            }

        elif model_choice == "transformer":
            custom_input_reshaped = custom_input_scaled.reshape(1, 1, custom_input_scaled.shape[1])
            prediction = transformer_model_loaded.predict(custom_input_reshaped)
            y_pred = transformer_model_loaded.predict(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))
            metrics = {
                "MSE": mean_squared_error(y_test, y_pred)
            }

        # Prepare the result to display on the webpage
        result = {
            "Selected Model": model_choice.upper(),
            "Prediction": prediction.tolist()[0] if hasattr(prediction, "tolist") else prediction[0],
            "Evaluation Metrics": metrics
        }

        # Check if classification report exists
        if "Classification Report" not in metrics:
            metrics["Classification Report"] = {}

        return render_template('results.html', result=result)

    except ValueError as e:
        return render_template('index.html', error="Invalid input, please ensure all fields are correctly filled.")


if __name__ == '__main__':
    app.run(debug=True)
