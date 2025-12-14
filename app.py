from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import pandas as pd
import os
from src.model_loader import load_model, predict

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    """Home page with project information and model comparison"""
    return render_template('index.html')

@app.route("/analysis")
def analysis():
    """Analysis page with CSV upload form"""
    return render_template('analysis.html')

@app.route("/reference")
def reference():
    """Reference page with citations and resources"""
    return render_template('reference.html')

@app.route("/acknowledgement")
def acknowledgement():
    """Acknowledgement page with team information and credits"""
    return render_template('acknowledgement.html')

@app.route("/predict", methods=['POST'])
def predict_csv():
    """
    Endpoint to accept CSV file and return HTML page with predictions dashboard and table.
    """
    try:
        # Check if file is present in request
        if 'file' not in request.files:
            return render_template('error.html', error="No file provided"), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return render_template('error.html', error="No file selected"), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return render_template('error.html', error="Invalid file type. Only CSV files are allowed"), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            
            if df.empty:
                return render_template('error.html', error="CSV file is empty"), 400
            
            # Load model and metadata
            model, metadata = load_model()
            
            # Make predictions
            predictions, probabilities, valid_indices = predict(model, df, metadata)
            
            # Get original data for valid rows
            df_valid = df.iloc[valid_indices].copy()
            
            # Calculate statistics
            total_records = len(predictions)
            attack_count = sum(1 for p in predictions if p == 'attack')
            normal_count = sum(1 for p in predictions if p == 'normal')
            
            attack_percentage = (attack_count / total_records * 100) if total_records > 0 else 0
            normal_percentage = (normal_count / total_records * 100) if total_records > 0 else 0
            
            # Prepare data for HTML table
            table_data = []
            for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
                row_data = {
                    'record_id': valid_indices[idx] + 1,
                    'prediction': pred,
                    'probability': round(prob * 100, 2),
                    'row_data': df_valid.iloc[idx].to_dict()
                }
                table_data.append(row_data)
            
            # Render HTML response
            return render_template(
                'results.html',
                filename=filename,
                total_records=total_records,
                attack_count=attack_count,
                normal_count=normal_count,
                attack_percentage=round(attack_percentage, 2),
                normal_percentage=round(normal_percentage, 2),
                table_data=table_data,
                df_columns=list(df_valid.columns)
            ), 200
            
        except pd.errors.EmptyDataError:
            return render_template('error.html', error="CSV file is empty or invalid"), 400
        except ValueError as e:
            return render_template('error.html', error=str(e)), 400
        except Exception as e:
            return render_template('error.html', error=f"Error processing CSV: {str(e)}"), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
    except Exception as e:
        return render_template('error.html', error=f"Server error: {str(e)}"), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)