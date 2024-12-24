import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_default_secret_key')  # Use a default secret key if not set
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the ResNet model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Define the base path for celebrity images
base_path = 'C:/Users/surface/celebrity_app/static/celebrities'

# Define a celebrity dictionary with absolute paths
celebrity_dict = {
    "Celebrity 1": os.path.join(base_path, 'celebrities','000001.jpg'),
    "Celebrity 2": os.path.join(base_path, 'celebrities','000002.jpg'),
    "Celebrity 3": os.path.join(base_path, 'celebrities','000003.jpg'),
    "Celebrity 4": os.path.join(base_path, 'celebrities','000004.jpg'),
    "Celebrity 5": os.path.join(base_path, 'celebrities','000005.jpg')
}

# Function to extract features
def extract_features(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = model.predict(img_array)
    return features

# Function to find celebrity look-alike
def find_celebrity_lookalike(target_image_path):
    target_features = extract_features(target_image_path)
    results = []
    
    for celebrity, image_path in celebrity_dict.items():
        celebrity_features = extract_features(image_path)
        
        # Calculate similarity
        similarity = np.dot(target_features, celebrity_features.T) / (np.linalg.norm(target_features) * np.linalg.norm(celebrity_features))
        
        # Convert similarity to a scalar value
        similarity_value = similarity.item()  # Use .item() to get a scalar
        
        results.append((celebrity, similarity_value, image_path))
    
    # Sort by similarity score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]  # Return top 5 look-alikes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash("No file part", "error")
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(url_for('index'))
    
    # Validate file type
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
        flash("Invalid file type. Please upload an image.", "error")
        return redirect(url_for('index'))
    
    # Save the file using the original filename (not recommended for security reasons)
    target_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(target_image_path)

    # Process the image and find look-alikes
    celebrities = find_celebrity_lookalike(target_image_path)

    # Optionally, clean up the uploaded image after processing
    # os.remove(target_image_path)

    return render_template('result.html', target_image=file.filename, celebrities=celebrities)

if __name__ == '__main__':
    app.run(debug=True)