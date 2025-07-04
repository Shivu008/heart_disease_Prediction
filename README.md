# 💓 Heart Disease Prediction Using Machine Learning
This project builds a machine learning model to predict whether a person has heart disease based on medical symptoms and diagnostic features. 
A synthetic dataset of 100,000 patient records is used to simulate real-world data, making it ideal for model training and evaluation.

# Project File
- `Heart_Disease_Prediction.ipynb` – Complete Jupyter Notebook with preprocessing, model building, evaluation, and feature analysis.
- `Heart_Disease_Patient_Data.csv` – Synthetic dataset (optional for sharing).
- `heart_disease_model.pkl` – Trained Random Forest model saved using joblib.
- `scaler.pkl` – StandardScaler object used for scaling input features.

# Features
- Synthetic dataset with 100,000 records
- Data preprocessing using encoding and scaling
- Random Forest Classifier used for training
- Performance evaluation using accuracy, confusion matrix, and classification report
- Feature importance analysis with visualization
- Model saved for future deployment

# Libraries Used
- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

# Model Download
The trained model file is hosted externally due to file size limits on GitHub.
🔗 Download heart_disease_model.pkl from Google Drive  (https://drive.google.com/uc?export=download&id=1A2B3C4D5E6F7G8H9I)

# How to Run the Project
1. Clone or download this repository.
2. Install the required libraries using:

   ```bash
   pip install -r requirements.txt

# Model Overview
Model Used: Random Forest Classifier
Dataset Size: 1,00,000 synthetic records
Accuracy: ~85% (varies slightly per run)
Target Variable: Target → 1 = Heart Disease Present, 0 = Not Present


