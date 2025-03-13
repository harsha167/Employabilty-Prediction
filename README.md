Employability Prediction with Gradio

Overview

A simple web application that predicts employability using machine learning and Gradio.

Prerequisites

Python 3.x

Gradio

scikit-learn

joblib

numpy

Installation

pip install -r requirements.txt

Usage

Ensure you have the model and label encoder files in the project directory:

employability_model_selected.joblib

label_encoder_fixed.joblib

Run the Gradio interface:

python app.py

Access the interface in your browser at:

http://localhost:7860

Code Overview

Main Function

The predict_employability function takes five inputs related to employability traits and predicts the status using a machine learning model.

Gradio Interface

A user-friendly web interface built with Gradio allows users to input ratings from 1 to 5 for each trait and get a prediction.

Hugging Face : https://huggingface.co/spaces/harsha167/Employability_Prediction

