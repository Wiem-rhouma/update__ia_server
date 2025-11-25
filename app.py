def preprocess_image(image_data):
    """Preprocess base64 image for your model"""
    try:
        # If it's a base64 string with header, remove header
        if isinstance(image_data, str) and 'base64,' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to match your model input
        image = image.resize(IMG_SIZE)  # (224, 224) or your model size
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return None
