import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from timm import create_model
from torchvision.models import vit_b_16
from huggingface_hub import hf_hub_download

# Classes
class_names = ['Aphids', 'Healthy', 'Mitesthrips']
num_classes = len(class_names)

# Remedies
remedies = {
    "Aphids": """ Remedy for Aphids:
- Spray neem oil or insecticidal soap on affected plants.
- Introduce natural predators like ladybugs or lacewings.
- Remove heavily infested leaves manually.
- Keep the area weed-free to prevent spread.""",

    "Mitesthrips": """ Remedy for Mitesthrips:
- Use reflective mulches to repel thrips.
- Apply insecticides or miticides as recommended.
- Encourage beneficial predators like predatory mites.
- Remove and destroy infected plant parts."""
}

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load model from Hugging Face
@st.cache_resource
def load_model(model_name):
    if model_name == "ViT-B/16 (torchvision)":
        model = vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        model_path = hf_hub_download(repo_id="your-username/chili-detection-models", filename="vit_b16_feature_extraction.pth")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    elif model_name == "ConViT (timm)":
        model = create_model("convit_base", pretrained=False)
        model.head = nn.Linear(model.head.in_features, num_classes)
        model_path = hf_hub_download(repo_id="your-username/chili-detection-models", filename="Convit_base.pth")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        raise ValueError("Unsupported model selected")

    model.eval()
    return model

# Streamlit App
st.title("Chilli Leaf Disease Classifier 🌿🌶️")
st.write("Upload an image and choose a model to classify the disease.")

model_option = st.selectbox("Choose Model", ["ViT-B/16 (torchvision)", "ConViT (timm)"])
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        try:
            model = load_model(model_option)
            input_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = class_names[predicted.item()]

            st.success(f"Prediction: **{prediction}** using **{model_option}**")

            # Save prediction to session_state
            st.session_state['prediction'] = prediction
            st.session_state['show_remedy'] = False  # Reset remedy flag

        except Exception as e:
            st.error(f"Error during prediction: {e}")

    # If a prediction was made
    if 'prediction' in st.session_state:
        prediction = st.session_state['prediction']

        # Show remedy button if needed
        if prediction in remedies:
            if st.button(f"Show Remedy for {prediction}"):
                st.session_state['show_remedy'] = True

        # If remedy button was clicked, show remedy
        if st.session_state.get('show_remedy', False):
            st.markdown(remedies[prediction])
