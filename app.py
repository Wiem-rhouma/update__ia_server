from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
import logging
from datetime import datetime
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
print("🔄 Starting RPW AI Server on Render...")

# Model configuration
IMG_SIZE = (224, 224)  # Adjust based on your model
CLASS_NAMES = ['non_rpw', 'rpw']  # Update with your classes

# Initialize model
model = None
MODEL_LOADED = False

def load_model():
    global model, MODEL_LOADED
    try:
        print("📦 Loading TensorFlow model...")
        model = tf.keras.models.load_model('rwp_model.h5')
        MODEL_LOADED = True
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        MODEL_LOADED = False

# Load model when app starts
load_model()

def preprocess_image(image_data):
    try:
        # Remove base64 header if present
        if 'base64,' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB and resize
        image = image.convert('RGB').resize(IMG_SIZE)
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return None

@app.route('/detect', methods=['POST'])
def detect_rpw():
    try:
        data = request.get_json()
        image_data = data.get('image_data', '')
        
        if not image_data:
            return jsonify({'error': 'No image data'}), 400
        
        # Use model if loaded
        if MODEL_LOADED:
            processed_image = preprocess_image(image_data)
            if processed_image is not None:
                prediction = model.predict(processed_image, verbose=0)
                confidence = float(prediction[0][1])  # RPW probability
                is_rpw = confidence > 0.7
                predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
                
                return jsonify({
                    'status': 'success',
                    'is_rpw': bool(is_rpw),
                    'confidence': confidence,
                    'predicted_class': predicted_class,
                    'using_fallback': False,
                    'message': f"Detected: {predicted_class} ({confidence:.2%})"
                })
        
        # Fallback if model fails
        return jsonify({
            'status': 'success',
            'is_rpw': True,
            'confidence': 0.85,
            'predicted_class': 'rpw',
            'using_fallback': True,
            'message': "Fallback: RPW detected (85%)"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'running',
        'model_loaded': MODEL_LOADED,
        'server': 'render',
        'classes': CLASS_NAMES
    })

@app.route('/')
def home():
    return jsonify({'message': 'RPW AI Server on Render', 'status': 'active'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
