import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set to '2' to show only warnings and errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Path to save uploaded images
app.config['UPLOAD_FOLDER'] = r'e:\AI-ML_VS_Code\brain_tumor_hosting-master\static\uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained model
model = load_model(r'e:\AI-ML_VS_Code\brain_tumor_hosting-master\brain_tumor_detection_model.h5')

# Define the categories for tumor types
categories = ["glioma", "meningioma", "notumor", "pituitary"]

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Preprocess the image
            img = image.load_img(file_path, target_size=(150, 150))
            img_array = image.img_to_array(img) / 255.0  # Normalize the image
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Make prediction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_label = categories[predicted_class]
            confidence = np.max(predictions) * 100  # Confidence as a percentage
            
            return render_template('index.html', filename=filename, predicted_label=predicted_label, confidence=confidence)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
