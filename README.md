# Facial Emotion Recognition System 🎭

An end-to-end Deep Learning web application for real-time facial emotion recognition. Built using **PyTorch (EfficientNet-B0)**, **OpenCV**, and **FastAPI**, with an interactive, responsive front-end supporting both single-image analysis and live camera detection.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Features

- **High-Accuracy Deep Learning Model**: Uses transfer learning with **EfficientNet-B0** pre-trained on ImageNet and fine-tuned on facial emotions (FER-2013 dataset), achieving over **70% validation accuracy**.
- **7 Recognized Emotions**:
  - 😠 Angry
  - 🤢 Disgust
  - 😨 Fear
  - 😄 Happy
  - 😐 Neutral
  - 😢 Sad
  - 😲 Surprise
- **Face Detection with Auto-Cropping**: Automatically detects faces using OpenCV Haar Cascades and crops with adaptive margins for focused classification.
- **Dual Input Modes**:
  - 📁 **Image Upload**: Drag & drop or browse photos (PNG, JPG, JPEG, WEBP).
  - 📷 **Live Webcam**: Real-time snapshot analysis from your browser camera.
- **Confidence Breakdown**: Returns the top predicted emotion, confidence percentage, and visual probability distribution across all 7 emotions.
- **RESTful API**: Fast and asynchronous inference endpoints with auto-generated Swagger documentation.

---

## 🏗️ Project Architecture

```text
├── best_efficientnet_b0 (1).pth  # Trained model weights (EfficientNet-B0)
├── main.py                       # FastAPI application & inference pipeline
├── requirements.txt              # Python dependencies
├── static/
│   └── index.html                # Modern glassmorphism UI (Tailwind CSS + JS)
├── .gitignore                    # Git ignore file for Python/venv
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/gangadharbodhapalle0199-stack/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### 3. Create and Activate Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note for CPU-only systems**: If you wish to install the lightweight CPU-only PyTorch build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 🏃 Running the Application

Start the FastAPI application with Uvicorn:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Once running, access:
- **Web Interface**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Documentation (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative API Documentation (ReDoc)**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔌 API Endpoints

### `POST /predict`
Upload an image file to get emotion predictions.

- **Request**: `multipart/form-data` with key `file` containing an image.
- **Response**:
```json
{
  "success": true,
  "prediction": "happy",
  "confidence": 0.942,
  "face_detected": true,
  "breakdown": {
    "angry": 0.002,
    "disgust": 0.001,
    "fear": 0.005,
    "happy": 0.942,
    "neutral": 0.035,
    "sad": 0.008,
    "surprise": 0.007
  }
}
```

---

## 🧠 Model Details

| Attribute | Detail |
|---|---|
| **Base Architecture** | EfficientNet-B0 |
| **Input Shape** | 224 × 224 × 3 (RGB) |
| **Normalization** | ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) |
| **Output Classes** | 7 classes (Softmax probability) |
| **Face Detection** | OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See [LICENSE](file:///c:/Users/bodha/OneDrive/Desktop/YOJAK/LICENSE) for more information.
