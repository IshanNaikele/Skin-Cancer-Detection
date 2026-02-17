from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI(title="DermaScan AI Backend")

# 1. Enable CORS (Critical for JS Frontend to communicate with FastAPI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the TFLite model (Ensure the file is in the same folder)
try:
    interpreter = tf.lite.Interpreter(model_path="skin_cancer_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error loading model: {e}")

def preprocess_for_tflite(image_data):
    # Convert bytes to OpenCV format
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # MATCH TRAINING: 128x128 resolution, NO dull-razor
    img = cv2.resize(img, (128, 128))
    
    # Normalize to [0, 1]
    img_array = img.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    # Read image contents
    contents = await file.read()
    input_data = preprocess_for_tflite(contents)

    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Map to Risk Categories
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    idx = np.argmax(output_data)
    prediction = classes[idx]
    confidence = float(output_data[0][idx])
    
    risk_map = {
        'mel': 'High Risk', 'bcc': 'High Risk', 'akiec': 'High Risk',
        'bkl': 'Moderate Risk', 'df': 'Moderate Risk', 'vasc': 'Moderate Risk',
        'nv': 'Low Risk'
    }
    
    return {
        "diagnosis": prediction,
        "risk_level": risk_map[prediction],
        "confidence": confidence
    }