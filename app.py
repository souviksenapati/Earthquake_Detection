from flask import Flask, request, render_template_string, jsonify
import pandas as pd
import numpy as np
import tempfile
import os
from tensorflow.keras.models import load_model
import json

app = Flask(__name__)

# Load the trained model
model = load_model("earthquake10_model.h5")

# HTML Templates for the Frontend
INDEX_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Earthquake Prediction</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(to right, #e0eafc, #cfdef3);
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .card {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            text-align: center;
            width: 90%;
            max-width: 500px;
        }
        h1 {
            margin-bottom: 20px;
            color: #333;
        }
        input[type=file] {
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 6px;
            width: 100%;
            margin-bottom: 20px;
        }
        button {
            background-color: #007BFF;
            color: white;
            padding: 12px 24px;
            font-size: 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        button:hover {
            background-color: #0056b3;
        }
        footer {
            margin-top: 30px;
            font-size: 12px;
            color: #777;
        }
    </style>
</head>
<body>
    <div class="card">
        <h1>Earthquake Prediction</h1>
        <form action="/upload" method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv" required>
            <br>
            <button type="submit">Upload and Predict</button>
        </form>
        <footer>Developed by Abhishek</footer>
    </div>
</body>
</html>
'''

RESULT_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f0f2f5;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .result-box {
            background-color: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
            text-align: center;
            width: 90%;
            max-width: 500px;
        }
        .prediction {
            font-size: 22px;
            margin: 20px 0;
            color: #333;
        }
        a {
            text-decoration: none;
            color: #007BFF;
            font-weight: bold;
        }
        a:hover {
            text-decoration: underline;
        }
        canvas {
            width: 100%;
            height: 300px;
        }
    </style>
</head>
<body>
    <div class="result-box">
        <h1>Prediction Result</h1>
        <p class="prediction">Predicted time to failure: <strong>{{ prediction }}</strong> seconds</p>
        <h3>Waveform of Acoustic Data</h3>
        <canvas id="waveform"></canvas>
        <a href="/">← Upload another file</a>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        var waveformData = {{ waveform | tojson }};
        var ctx = document.getElementById('waveform').getContext('2d');
        var chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: waveformData.map((_, index) => index),
                datasets: [{
                    label: 'Acoustic Data',
                    data: waveformData,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                scales: {
                    x: { 
                        title: { display: true, text: 'Index' }
                    },
                    y: { 
                        title: { display: true, text: 'Acoustic Data Value' }
                    }
                }
            }
        });
    </script>
</body>
</html>
'''

# Routes for the Flask app
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        try:
            # Save file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                file.save(tmp.name)
                tmp_path = tmp.name

            chunk_size = 20000
            reader = pd.read_csv(tmp_path, chunksize=chunk_size)

            X_test_new = []
            y_test_new = []

            # Read and process the chunks from CSV
            for chunk in reader:
                if len(chunk) < chunk_size:
                    continue

                # Normalize the acoustic data
                x = chunk['acoustic_data'].values.astype(np.float32)
                x = (x - x.mean()) / (x.std() + 1e-6)
                X_test_new.append(x.reshape(chunk_size, 1))

                # Collect the last time_to_failure value
                if 'time_to_failure' in chunk.columns:
                    y = chunk['time_to_failure'].values[-1]
                    y_test_new.append(y)

            # Convert lists to numpy arrays
            X_test_new = np.array(X_test_new, dtype=np.float32)
            y_test_new = np.array(y_test_new, dtype=np.float32) if y_test_new else None 

            # Make predictions
            y_pred = model.predict(X_test_new)
            predicted_value = float(y_pred[0][0])

            # Send data for visualization
            return render_template_string(
                RESULT_HTML,
                prediction=round(predicted_value, 6),
                waveform=x.tolist()  # Acoustic data for the graph
            )

        except Exception as e:
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        return jsonify({"error": "Invalid file type. Only .csv files are allowed."}), 400

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
