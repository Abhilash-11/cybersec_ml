from flask import Flask, request, jsonify
import joblib
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import pandas as pd
import os

# ===== initialize the flask app =====
app = Flask(__name__)

# ===== load the models =====
# load the machine learning models
ml_models_dir = 'models/ml_models'
ml_models = {}
for file in os.listdir(ml_models_dir):
    if file.endswith('.joblib'):
        model_name = file.split('.')[0]
        ml_models[model_name] = joblib.load(os.path.join(ml_models_dir, file))

# load the deep learning model
dl_model = load_model('models/dl_model/attn_model.keras')

# load the encoder file
encoder = joblib.load('models/encoder.joblib')

# load the feature colums order
feature_columns = joblib.load('models/feature_columns_selected.joblib')

# load selector and scaler
selector = joblib.load('models/selector.joblib')
scaler = joblib.load('models/scaler.joblib')

# ===== preprocessing helper function =====
def preprocess_input(data, model_type='ml'):
    """
    converts JSON input into a DataFrame in the correct order for the model
    :param data: data
    :param model_type: ml or dl model
    :return: df
    """

    # convert data to df
    df = pd.DataFrame([data])

    # replace missing data with 0
    for column in feature_columns:
        if column not in df.columns:
            df[column] = 0

    # ensure columns are in correct order
    df = df[feature_columns]

    # force numeric and handle infinities
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    # reshape for DL
    if model_type == 'dl':
        return df.values[..., np.newaxis]
    else:
        return df.values


# ===== ROUTES =====
# ===== ml prediction route =====
@app.route('/predict/ml/<model_name>', methods=['POST'])
def predict_ml(model_name):
    if model_name not in ml_models:
        return jsonify({"error": f"Model {model_name} not found"}), 404

    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # preprocess input
    X = preprocess_input(data, model_type='ml')
    # predict
    y_pred = ml_models[model_name].predict(X)
    label_pred = encoder.inverse_transform(y_pred)[0]

    return jsonify({"prediction": label_pred}), 200

# ===== deep learning prediction route =====
@app.route('/predict/dl', methods=['POST'])
def predict_dl():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data received'}), 400

    try:
        json_to_feature = {
            "Src_Port": "Src Port",
            "Dst_Port": "Dst Port",
            "Flow_Duration": "Flow Duration",
        }

        X = preprocess_input(data, model_type='dl')
        y_probs = dl_model.predict(X)
        y_pred = np.argmax(y_probs, axis=1)
        label_pred = encoder.inverse_transform(y_pred)

        # Single row vs batch
        if len(label_pred) == 1:
            label_pred = label_pred[0]
            y_probs = y_probs[0].tolist()
        else:
            label_pred = label_pred.tolist()
            y_probs = y_probs.tolist()

        return jsonify({'predicted_label': label_pred, 'probabilities': y_probs}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/schema', methods=['GET'])
def schema():
    all_feature_columns = joblib.load('models/feature_columns_selected.joblib')
    example = {name: 0 for name in all_feature_columns}
    # optional: put typical/example values for a few keys
    if "Src Port" in example:
        example["Src Port"] = 443
    if "Dst Port" in example:
        example["Dst Port"] = 80
    # return both the required field list and a JSON skeleton
    return jsonify({
        "n_required_features": len(all_feature_columns),
        "features": all_feature_columns,
        "example_request": example
    }), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)