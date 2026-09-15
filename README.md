# Deep Learning-Based Pneumonia Detection from Chest X-Rays

## Project Overview

A deep learning-based computer vision system for classifying pediatric chest X-ray images as Normal or Pneumonia.

The project uses EfficientNetB0 transfer learning and Grad-CAM explainability to provide predictions and visualize image regions contributing to the prediction.

## Dataset

- Pediatric Chest X-ray dataset
- Classes: NORMAL and PNEUMONIA
- Training images: 5216
- Test images: 624
- Validation: 15% split from training data

## Methodology

1. Load and organize chest X-ray images.
2. Resize images to 224 x 224.
3. Apply data augmentation.
4. Use ImageNet-pretrained EfficientNetB0.
5. Add Global Average Pooling and fully connected layers.
6. Train using binary cross-entropy.
7. Evaluate using Accuracy, Precision, Recall, F1 Score and ROC-AUC.
8. Use a classification threshold of 0.90.
9. Generate Grad-CAM visualizations.
10. Deploy using Streamlit.

## Model Architecture

Input X-ray (224 x 224 x 3)
-> EfficientNetB0
-> Global Average Pooling
-> Dropout
-> Dense (128, ReLU)
-> Dropout
-> Sigmoid Output
-> NORMAL / PNEUMONIA

## Performance

| Metric | Score |
|---|---:|
| Accuracy | 86.54% |
| Precision | 85.25% |
| Recall | 94.87% |
| F1 Score | 89.81% |
| ROC-AUC | 94.47% |

## Explainability

Grad-CAM is used to visualize image regions that contribute strongly to the pneumonia prediction.

## Application

The Streamlit application allows users to:

- Upload a chest X-ray
- Obtain Normal/Pneumonia prediction
- View prediction confidence
- View Grad-CAM heatmap
- View Grad-CAM overlay

## Technologies

- Python
- TensorFlow / Keras
- EfficientNetB0
- Deep Learning
- Computer Vision
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- Grad-CAM

## Project Structure

chest_xray_project/
|
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- models/
|   |-- chest_xray_pneumonia_efficientnetb0.keras
|
|-- results/
    |-- evaluation_metrics.csv
    |-- gradcam/
        |-- pneumonia_gradcam_example.png

## Disclaimer

This project is intended for educational and research purposes. The predictions are not medical diagnoses and should not replace professional medical evaluation.
