# 🩺 AI-Powered Gastrointestinal Disease Detection

A deep-learning web application for classifying gastrointestinal endoscopy images using a fine-tuned **VGG19** model.

The application is built with **Django** and uses an optimized **LiteRT / TensorFlow Lite model** for lightweight inference and free cloud deployment.

## 🌐 Live Demo

👉 https://ulcer-detection-vgg19.onrender.com

> The free Render instance may take a short time to wake up after a period of inactivity.

---

## 📌 Project Overview

This project was developed as an academic final-year project to demonstrate the use of deep learning for gastrointestinal endoscopy image classification.

Users can upload an endoscopy image through the web interface. The application preprocesses the image and sends it to the trained AI model, which predicts one of four supported gastrointestinal image classes.

### Supported Classes

- Normal
- Esophagitis
- Ulcerative Colitis
- Polyps

---

## 🤖 Model

The project uses **VGG19 Transfer Learning** with ImageNet pretrained weights.

The initial VGG19 feature extractor was frozen and trained with a custom classification head. Selected upper VGG19 layers were then fine-tuned using a low learning rate.

### Deployment Model Performance

| Metric | Result |
|---|---:|
| Test Accuracy | **85.17%** |
| Weighted Precision | **85.51%** |
| Weighted Recall | **85.17%** |
| Weighted F1-Score | **85.27%** |

### Per-Class Test Performance

| Class | Accuracy |
|---|---:|
| Normal | 81.33% |
| Esophagitis | 80.67% |
| Ulcerative Colitis | 91.33% |
| Polyps | 87.33% |

The final optimized model is approximately **19.32 MB** and is used by the deployed Django application through LiteRT.

---

## 🛠 Technology Stack

### Backend

- Python 3.11
- Django 5.2
- Gunicorn
- WhiteNoise

### Machine Learning

- VGG19
- Transfer Learning
- Fine-Tuning
- LiteRT / TensorFlow Lite
- NumPy
- Pillow

### Frontend

- HTML5
- CSS3
- JavaScript
- Responsive Design

### Deployment

- GitHub
- Render

---

## 🔄 Application Workflow

```text
Endoscopy Image Upload
        ↓
Image Validation
        ↓
RGB Conversion
        ↓
Resize to 112 × 112
        ↓
Optimized VGG19 LiteRT Model
        ↓
4-Class Probability Prediction
        ↓
Predicted Disease Class
        ↓
Confidence Score
        ↓
Web Result Display 

```
---

##  🧠 Model Development Workflow

```text
Kvasir V2 Dataset
        ↓
4-Class Dataset Preparation
        ↓
Balanced Dataset
        ↓
Train / Validation / Test Split
        ↓
VGG19 Transfer Learning
        ↓
Fine-Tuning
        ↓
Model Evaluation
        ↓
TensorFlow Lite Conversion
        ↓
Optimized LiteRT Evaluation
        ↓
Django Integration
        ↓
Render Deployment
```
---
## 📂 Project Structure

ulcer-detection-vgg19/
│
├── detector/
│   ├── services/
│   │   └── predictor.py
│   │
│   ├── static/
│   │   └── detector/
│   │       ├── css/
│   │       ├── images/
│   │       └── js/
│   │
│   ├── templates/
│   │   └── detector/
│   │       └── upload.html
│   │
│   ├── forms.py
│   ├── urls.py
│   └── views.py
│
├── model/
│   ├── class_names.json
│   └── vgg19_ulcer_disease_finetuned_optimized.tflite
│
├── training/
│   ├── prepare_dataset.py
│   ├── split_dataset.py
│   ├── train_model.py
│   ├── fine_tune_model.py
│   ├── evaluate_model.py
│   ├── evaluate_finetuned_model.py
│   ├── convert_model_to_tflite.py
│   └── evaluate_tflite_model.py
│
├── ulcer_detection/
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
│
├── .gitignore
├── .python-version
├── manage.py
├── requirements.txt
└── README.md
---

## ☁️ Deployment

The application is deployed on Render using:

- Python 3.11
- Django 5.2
- Gunicorn
- WhiteNoise
- LiteRT
- Optimized 19.32 MB TFLite model
---
## ⚠️ Model Limitations

The model is a closed-set four-class image classifier.

It can classify only:

- Normal
- Esophagitis
- Ulcerative Colitis
- Polyps

Images outside these categories may still be assigned to one of the four supported classes.

The model achieved **85.17% test accuracy** on the prepared test dataset and should not be considered a clinical diagnostic system.

---
