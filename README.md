# Employability Prediction with Gradio

## Overview
A simple web application that predicts employability using machine learning and Gradio.

## Prerequisites
- Python 3.x
- Gradio
- scikit-learn
- joblib
- numpy

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Ensure you have the model and label encoder files in the project directory:
   - `employability_model_selected.joblib`
   - `label_encoder_fixed.joblib`

2. Run the Gradio interface:
```bash
python app.py
```

## How It Works
1. **Input Data**: Users provide ratings from 1 to 5 on five attributes: Manner of Speaking, Self-Confidence, Ability to Present Ideas, Communication Skills, and Mental Alertness.
2. **Model Prediction**: The input data is passed to a pre-trained machine learning model loaded using joblib.
3. **Encoding Handling**: The model outputs a numerical prediction, which is decoded back to a human-readable employability status using a label encoder.
4. **Result Display**: The result is displayed with a positive or negative emoji to indicate employability status.

## Code Overview

### Main Function
The `predict_employability` function takes five inputs related to employability traits and predicts the status using a machine learning model.

### Gradio Interface
A user-friendly web interface built with Gradio allows users to input ratings from 1 to 5 for each trait and get a prediction.

