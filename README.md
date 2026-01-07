# Network Intrusion Detection API (Flask + ML + DL)

A Flask-based REST API that serves both **Machine Learning (ML)** and **Deep Learning (DL)** models for **network intrusion detection**.
It supports multiple ML models (loaded dynamically), a deep learning attention-based model, automatic preprocessing (feature alignment, selection, scaling), and JSON-based inference.

---

## Features

**Supports Multiple Models:** Dynamically loads all `.joblib` ML models.
**Deep Learning Integration:** Loads a Keras attention-based model.
* **Automatic Preprocessing:**

  * Reconstructs missing features.
  * Applies trained `selector` and `scaler`.
  * Handles partial JSON safely.
**Prediction Outputs:**

  * ML → predicted labels
  * DL → predicted labels + class probabilities
**Reusable Components:**

  * Modular preprocessing function.
  * JSON-to-feature mapping for friendly input keys.

---

## Project Structure

```
project/
│
├── app.py                             
│
├── models/
│   ├── ml_models/                     
│   │   ├── SDG.joblib
│   │   ├── RandomForest.joblib
│   │   └── NaiveBayes.jolib
|   |    
│   │
│   ├── dl_model/
│   │   └── attn_model.keras           
│   │
│   ├── encoder.joblib                 
│   ├── selector.joblib                
│   ├── scaler.joblib  
│   └── feature_columns_selected.joblib
│
├── network-intrusion-detection.ipynb
├── README.md
└── requirements.txt
```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Abhilash-11/cybersec_ml.git
cd cybersec_ml
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Start the Flask API

```bash
python app.py
```

By default, it will run at:

```
http://0.0.0.0:5000
```

---

## Available Endpoints

### **1. Machine Learning Prediction**

```
POST /predict/ml/<model_name>
```

#### Parameters

| Name         | Type   | Description                                     |
| ------------ | ------ | ----------------------------------------------- |
| `model_name` | string | Name of the loaded ML model (without `.joblib`) |
| **Body**     | JSON   | Input feature(s) as key-value pairs             |

---

### **2. Deep Learning Prediction**

```
POST /predict/dl
```


#### Response

Returns both predicted class label and probability distribution.

---
### **3. Schema for the request design**
```commandline
POST /schema
```
#### Response
Return an object whose first property is an object containing the schema of the example of features of the requests made.

---

## Example Request (Postman)

### URL

```
POST http://localhost:5000/predict/ml/SGD
```

### Headers

```
Content-Type: application/json
```

### Example Body

```json
{
  "Src Port": 443,
  "Dst Port": 51234,
  "Flow Duration": 12500,
  "Tot Fwd Pkts": 20,
  "Tot Bwd Pkts": 30,
  "TotLen Fwd Pkts": 14000,
  "TotLen Bwd Pkts": 21000,
  "Fwd Pkt Len Max": 700,
  "Bwd Pkt Len Max": 750,
  "Flow Byts/s": 3200,
  "Flow Pkts/s": 8.5,
  "Fwd Seg Size Avg": 500
}
```

> You can send **partial JSON** — missing features will automatically be filled with `0`.

### Example Response

```json
{
  "prediction": "BENIGN"
}
```

---

### Deep Learning Request Example

#### URL

```
POST http://localhost:5000/predict/dl
```

#### Body

```json
{
  "Src_Port": 443,
  "Dst_Port": 51234,
  "Flow_Duration": 12500,
  "Tot Fwd Pkts": 25,
  "Tot Bwd Pkts": 35,
  "TotLen Fwd Pkts": 15000,
  "TotLen Bwd Pkts": 21000
}
```

#### Response

```json
{
  "predicted_label": "BENIGN",
  "probabilities": [0.92, 0.03, 0.05]
}
```

---

## How the Preprocessing Works

The API automatically ensures that the input features match those used during model training:

1. **Builds DataFrame** from incoming JSON.
2. **Maps JSON keys** to training feature names if a mapping dict is defined.
3. **Adds missing features** (sets them to zero).
4. **Reorders columns** exactly as during model training.
5. **Applies selector & scaler** for ML models.
6. **Reshapes input** for DL models.

---

## Example `.env` (optional)

If you want configurable ports or debug mode:

```
FLASK_ENV=development
FLASK_RUN_PORT=5000
```

Then run:

```bash
flask run
```

---
