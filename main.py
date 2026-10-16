import io
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# 1. FER-2013 classes in alphabetical order
EMOTION_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# 2. FastAPI app & CORS Setup
app = FastAPI(title="Facial Emotion Recognition API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Model Architecture & Weight Loading
device = torch.device("cpu")

# Support both EfficientNet-B0 (current higher accuracy model) and ResNet18
BASE_DIR = Path(__file__).resolve().parent

# Check for model file
candidate_models = [
    BASE_DIR / "best_efficientnet_b0 (1).pth",
    BASE_DIR / "best_efficientnet_b0.pth",
    BASE_DIR / "model" / "best_efficientnet_b0.pth",
    BASE_DIR / "best_resnet_emotion.pth",
    BASE_DIR / "model" / "best_resnet_emotion.pth",
    BASE_DIR / "model" / "best_emotion_model.pth",
]

MODEL_PATH = None
for p in candidate_models:
    if p.exists():
        MODEL_PATH = p
        break

model = None
model_arch = "efficientnet_b0"

if MODEL_PATH and MODEL_PATH.exists():
    try:
        ckpt = torch.load(str(MODEL_PATH), map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            model_arch = ckpt.get("architecture", "efficientnet_b0")
            if "classes" in ckpt and isinstance(ckpt["classes"], list):
                EMOTION_CLASSES = ckpt["classes"]
        else:
            state_dict = ckpt

        # Check if state_dict has EfficientNet or ResNet keys
        is_efficientnet = any("features" in k for k in state_dict.keys())

        if is_efficientnet or model_arch == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(1280, len(EMOTION_CLASSES))
            model.load_state_dict(state_dict)
            model_arch = "efficientnet_b0"
            print(f"Loaded EfficientNet-B0 model weights successfully from: {MODEL_PATH}")
        else:
            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_ftrs, len(EMOTION_CLASSES)))
            model.load_state_dict(state_dict)
            model_arch = "resnet18"
            print(f"Loaded ResNet18 model weights successfully from: {MODEL_PATH}")

    except Exception as e:
        print(f"Warning: Could not load state_dict directly ({e}). Initializing fallback EfficientNet-B0.")
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(1280, len(EMOTION_CLASSES))
else:
    print(f"Warning: Model file not found. Initializing fallback EfficientNet-B0.")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, len(EMOTION_CLASSES))

model.to(device)
model.eval()

# 4. Face Detection Setup (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 5. Preprocessing Pipeline
# EfficientNet-B0 expects 3-channel RGB, standard 224x224 input and ImageNet normalization
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# 6. Endpoints
@app.get("/")
async def serve_index():
    index_file = BASE_DIR / "static" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_file)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        # Read uploaded image bytes
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image content.")

        height, width = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Detect faces using OpenCV Haar Cascade
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        face_detected = len(faces) > 0

        if face_detected:
            # Select the primary (largest area) face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            # Apply 10% bounding box margin to eliminate background noise
            margin_x = int(0.10 * w)
            margin_y = int(0.10 * h)

            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(width, x + w + margin_x)
            y2 = min(height, y + h + margin_y)

            cropped = img_bgr[y1:y2, x1:x2]
        else:
            # Fallback to the entire image
            cropped = img_bgr

        # Convert OpenCV BGR to RGB PIL Image for torchvision transforms
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(cropped_rgb)

        # Preprocess for model inference (224x224 RGB ImageNet normalized)
        input_tensor = image_transform(pil_image).unsqueeze(0).to(device)

        # Model Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze(0).tolist()

        max_idx = int(np.argmax(probabilities))
        prediction = EMOTION_CLASSES[max_idx]
        confidence = float(probabilities[max_idx])

        breakdown: Dict[str, float] = {
            emotion: float(prob)
            for emotion, prob in zip(EMOTION_CLASSES, probabilities)
        }

        return {
            "success": True,
            "prediction": prediction,
            "confidence": confidence,
            "face_detected": face_detected,
            "breakdown": breakdown,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# 7. Mount Static Directory
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
