# NIDS Backend API

Flask backend for Network Intrusion Detection System (NIDS) that processes CSV files and returns predictions.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your trained model file in the backend directory:
   - Name it `model.pkl` or `model.joblib` (or update `MODEL_PATH` in `model_loader.py`)
   - Supported formats: pickle, joblib, or TensorFlow/Keras models

3. Update `model_loader.py`:
   - Modify `load_model()` to match your model format
   - Update `predict()` to preprocess your CSV data correctly
   - Adjust feature selection and preprocessing steps

4. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST /predict

Upload a CSV file and get predictions for each record.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Form data with key `file` containing the CSV file

**Example using curl:**
```bash
curl -X POST -F "file=@your_data.csv" http://localhost:5000/predict
```

**Example using Python requests:**
```python
import requests

url = "http://localhost:5000/predict"
files = {'file': open('your_data.csv', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

**Response:**
```json
{
  "total_records": 100,
  "predictions": [
    {
      "record_id": 1,
      "prediction": "normal"
    },
    {
      "record_id": 2,
      "prediction": "attack"
    }
    // ... more records
  ],
  "statistics": {
    "attack": {
      "count": 25,
      "percentage": 25.0
    },
    "normal": {
      "count": 75,
      "percentage": 75.0
    }
  }
}
```

## Model Integration

To integrate your model:

1. **Update `MODEL_PATH`** in `model_loader.py` to point to your model file

2. **Modify `load_model()`** function based on your model type:
   - For pickle: `pickle.load(open(MODEL_PATH, 'rb'))`
   - For joblib: `joblib.load(MODEL_PATH)`
   - For Keras: `keras.models.load_model(MODEL_PATH)`

3. **Update `predict()`** function:
   - Select the correct feature columns from your CSV
   - Apply the same preprocessing used during training (scaling, encoding, etc.)
   - Ensure predictions are converted to 'normal' or 'attack'

## Error Handling

The API handles:
- Missing files
- Invalid file types
- Empty CSV files
- Model loading errors
- Prediction errors

All errors return appropriate HTTP status codes with error messages.

"# netlog-analyst" 
