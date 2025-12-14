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

### Option 1: Using run.py (Recommended for auto-reload)
```bash
python run.py
```

### Option 2: Using app.py directly
```bash
python app.py
```

### Option 3: Using Flask CLI (Best for auto-reload)
```bash
# Set environment variables
set FLASK_APP=app.py
set FLASK_ENV=development
set FLASK_DEBUG=1

# Run Flask
flask run --host=0.0.0.0 --port=5000 --reload
```

The server will start on `http://localhost:5000`

## Auto-Reload

The server is configured with auto-reload enabled. When you make changes to:
- Python files (`.py`)
- Template files (`.html`)
- Static files (`.css`, `.js`)

The server will automatically restart. You should see messages like:
```
 * Detected change in 'app.py', reloading
 * Restarting with stat
```

If auto-reload doesn't work:
1. Use `run.py` instead of `app.py`
2. Or use Flask CLI with `--reload` flag
3. Manually restart by pressing `Ctrl+C` and running again

## API Endpoints

### GET /
Home page with project information and model comparison.

### GET /analysis
Analysis page with CSV file upload form.

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

**Response:**
HTML page with dashboard showing:
- Total records processed
- Attack vs Normal statistics
- Detailed table with predictions for each record
- Searchable and paginated results

### GET /reference
Reference page with citations and research papers.

### GET /acknowledgement
Acknowledgement page with team information.

## Model Integration

The system uses Random Forest model trained on CICIDS-2017 dataset:
- Model file: `models/rf_cicids2017_model.pkl`
- Metadata file: `models/rf_cicids2017_metadata.pkl`

## File Structure

```
backend/
├── app.py                 # Main Flask application
├── run.py                 # Development server with auto-reload
├── src/
│   ├── __init__.py
│   └── model_loader.py    # Model loading and prediction functions
├── templates/
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Home page
│   ├── analysis.html     # Analysis/upload page
│   ├── results.html      # Results page with dashboard
│   ├── reference.html    # References page
│   ├── acknowledgement.html  # Acknowledgement page
│   └── error.html        # Error page
├── models/               # Model files
├── uploads/              # Temporary upload folder
└── static/               # Static files (CSS, JS, images)
```

## Error Handling

The API handles:
- Missing files
- Invalid file types
- Empty CSV files
- Model loading errors
- Prediction errors

All errors return appropriate HTTP status codes with error messages.
