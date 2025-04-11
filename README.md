# 🧠 Brain Tumour Detection System

## 📸 Screenshots
# 
<img src= "https://github.com/sahkanu34/Brain_tumour_detection_system/blob/main/screenshots/home.png?raw=true" >
<img src= "https://github.com/sahkanu34/Brain_tumour_detection_system/blob/main/screenshots/upload.png?raw=true" >
<img src= "https://github.com/sahkanu34/Brain_tumour_detection_system/blob/main/screenshots/about.png?raw=true" >
<img src= "https://github.com/sahkanu34/Brain_tumour_detection_system/blob/main/screenshots/results.png?raw=true" >

---
# Checkout the Live Demo app
[!Streamlit] (https://braintumourdetectionsystem-34.streamlit.app/)


# 🧠 Streamlit Deployment Instructions

This document provides step-by-step instructions for deploying your Brain Tumor Classification application on Streamlit Cloud.

## 📁 Project Structure

Your project should have the following structure:
```
brain-tumor-classifier/
├── .streamlit/
│   └── config.toml
├── models/
│   └── brain_tumor_classifier.h5
├── app.py
├── requirements.txt
├── runtime.txt
├── notebooks/
│   └── model_training.ipynb
```

## 🚀 Deployment Steps

### 1. Prepare Your GitHub Repository

1. Create a new GitHub repository (e.g., `brain-tumor-classifier`)
2. Upload all the files in this project to your repository:
    - `app.py` (your main Streamlit application)
    - `requirements.txt` (dependencies with specific versions)
    - `runtime.txt` (specifies Python 3.10.12)
    - `notebooks/model_training.ipynb` (train and export the model)
    - `.streamlit/config.toml` (theme configuration)

### 2. Set Up the Model

Before deploying, you have two options for handling the model:

**Option 1: Train using the notebook (recommended)**
- Use `model_training.ipynb` to train and export your model
- Follow the notebook instructions to create your classifier
- Save the model as `brain_tumor_classifier.h5`

**Option 2: Use your pre-trained model**
- Upload your existing trained model to the `models/` directory
- Name it `brain_tumor_classifier.h5` to match the path in the code

### 3. Deploy on Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch, and main file path (`app.py`)
5. Advanced Settings:
    - Python version: 3.10.12 (already specified in runtime.txt)
    - Add any required secrets if your app needs them
6. Click "Deploy"

### 4. First-time Setup

When your app is deployed for the first time:

1. Ensure your trained model is in place before deployment
2. The first load might take a few minutes as TensorFlow and other dependencies are installed
3. Once loaded, your app will be available at the provided Streamlit URL

## ⚠️ Important Notes

1. **Python Version**: Your app will use Python 3.10.12 as specified in the runtime.txt file
    - We've downgraded from 3.12.6 to 3.10.12 for TensorFlow compatibility
    - TensorFlow has known compatibility issues with Python 3.12

2. **TensorFlow Version**: We're using TensorFlow 2.10.0 specifically
    - This version is stable and compatible with Python 3.10
    - We've also pinned protobuf to 3.20.0 to avoid dependency conflicts

3. **Model File**: The app expects a model file at `models/brain_tumor_classifier.h5`

4. **Memory Usage**: TensorFlow can be memory-intensive, so your app might need the "Medium" resource tier on Streamlit Cloud

## 🔧 Troubleshooting

If you encounter issues during deployment:

1. Check the Streamlit Cloud logs for error messages

2. TensorFlow Errors:
    - If you see TensorFlow errors, check that you're using Python 3.10.12 and TensorFlow 2.10.0
    - Some newer TensorFlow versions may have compatibility issues with certain Python versions

3. Model Loading Errors:
    - Ensure the model file exists at the correct path
    - Check that the model format is compatible with the TensorFlow version

4. Memory Issues:
    - If the app crashes due to memory limits, try upgrading the resource tier in Streamlit Cloud settings

## 💻 Local Testing

Before deploying, you can test your app locally:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train model using Jupyter notebook
jupyter notebook notebooks/model_training.ipynb

# Run the Streamlit app
streamlit run app.py
```

This will allow you to verify everything works correctly before deploying to Streamlit Cloud.
