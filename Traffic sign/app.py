# gradio_app.py
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import os

# Try to import TensorFlow with error handling
try:
    import tensorflow as tf
    print("TensorFlow imported successfully!")
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"TensorFlow import error: {e}")
    print("Please install TensorFlow: pip install tensorflow")
    TENSORFLOW_AVAILABLE = False

# Class names from your notebook
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Veh > 3.5 tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}

def predict_traffic_sign(image):
    if not TENSORFLOW_AVAILABLE:
        return "Error: TensorFlow not installed. Please run: pip install tensorflow"
    
    try:
        # Load model
        if not os.path.exists('traffic_sign_model.h5'):
            return "Error: Model file 'traffic_sign_model.h5' not found. Please train and save the model first."
        
        model = tf.keras.models.load_model('traffic_sign_model.h5')
        
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Preprocess image (same as your training)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (64, 64))
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Predict
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return f"Prediction: {classes[predicted_class]}\nConfidence: {confidence:.2%}"
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Create interface without examples
iface = gr.Interface(
    fn=predict_traffic_sign,
    inputs=gr.Image(type="numpy", label="Upload Traffic Sign Image"),
    outputs="text",
    title="🚦 Traffic Sign Detection",
    description="Upload an image of a traffic sign to classify it into one of 43 categories"
)

if __name__ == "__main__":
    print("Launching Traffic Sign Detection App...")
    iface.launch(share=True)  # share=True creates a public link
    