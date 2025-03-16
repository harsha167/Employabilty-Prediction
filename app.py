import joblib
import gradio as gr
import numpy as np
import os

# Define model and encoder paths
MODEL_PATH = os.path.join(os.getcwd(), "employability_model_selected.joblib")
ENCODER_PATH = os.path.join(os.getcwd(), "label_encoder_fixed.joblib")

# Load model and encoder with error handling
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    model, label_encoder = None, None
    print(f"Error loading model: {e}")

def predict_employability(name, manner_of_speaking, self_confidence, ability_to_present_ideas, communication_skills, mental_alertness):
    """Predict employability based on user input."""
    if model is None or label_encoder is None:
        return "❌ Model not loaded. Please check the file paths."
    
    input_data = np.array([[manner_of_speaking, self_confidence, ability_to_present_ideas, communication_skills, mental_alertness]])
    prediction = model.predict(input_data)[0]
    result = label_encoder.inverse_transform([prediction])[0]

    # Display result with emojis and personalized message
    if result == "Employable":
        return f"✅ Congratulations, {name}! You are Employable."
    else:
        return f"😞 Sorry, {name}. You are Not Employable. Keep improving!"

# Gradio UI with a Name field and styled output box
iface = gr.Interface(
    fn=predict_employability,
    inputs=[
        gr.Textbox(label="Your Name", placeholder="Enter your name here"),
        gr.Slider(1, 5, step=1, label="Manner of Speaking", info="Rate your clarity and tone in speech."),
        gr.Slider(1, 5, step=1, label="Self-Confidence", info="Rate your confidence in professional settings."),
        gr.Slider(1, 5, step=1, label="Ability to Present Ideas", info="Rate how well you can articulate ideas."),
        gr.Slider(1, 5, step=1, label="Communication Skills", info="Rate your overall communication effectiveness."),
        gr.Slider(1, 5, step=1, label="Mental Alertness", info="Rate your ability to think quickly and adapt.")
    ],
    outputs=gr.Textbox(label="Prediction", interactive=False),
    title="Employability Prediction",
    description="Enter your name and rate yourself on the given attributes (1-5) to check your employability status.",
    theme="compact"
)

if __name__ == "__main__":
    iface.launch(share=True)
